"""Manual Tool Calling Executor - Bypasses LangChain's create_openai_tools_agent for Python 3.14 compatibility."""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseLanguageModel

logger = logging.getLogger(__name__)

METADATA_BLACKLIST = {"ROUND", "AGENT", "TURN", "STEP", "USER", "SYSTEM", "OTHER", "SA"}
RE_B3_TICKER = re.compile(r"^[A-Z]{4}(?:3|4|5|6|11)(?:\.SA)?$")
RE_YAHOO_NYSE_TICKER = re.compile(r"^[A-Z]{1,5}$")


class ManualToolCallingExecutor:
    """
    Manual tool calling executor that uses llm.bind_tools() to avoid
    Python 3.14 compatibility issues with LangChain's create_openai_tools_agent.
    
    This executor:
    1. Binds tools directly to the LLM
    2. Handles tool calls and executions manually
    3. Maintains conversation history
    4. Supports fallback to simple wrapper if bind_tools fails
    """
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        tools: List[BaseTool],
        system_prompt: str,
        max_iterations: int = 10,
        verbose: bool = True
    ):
        """
        Initialize the manual tool calling executor.
        
        Args:
            llm: The language model to use
            tools: List of tools available to the agent
            system_prompt: System prompt for the agent
            max_iterations: Maximum number of tool calling iterations
            verbose: Whether to log debug information
        """
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # Create tool map for easy lookup
        self.tool_map: Dict[str, BaseTool] = {tool.name: tool for tool in tools}
        
        # Try to bind tools directly to LLM
        self.llm_with_tools = self._try_bind_tools()
        
        logger.info(
            "ManualToolCallingExecutor initialized with %d tools (bind_tools: %s)",
            len(self.tools),
            self.llm_with_tools is not None
        )
    
    def _try_bind_tools(self) -> Optional[Any]:
        """
        Try to bind tools to the LLM using bind_tools() method.
        
        Returns:
            LLM with tools bound, or None if binding fails
        """
        try:
            # Attempt to use bind_tools if available (OpenAI, DeepSeek, etc.)
            if hasattr(self.llm, "bind_tools"):
                logger.info("Attempting to bind tools to LLM using bind_tools()")
                
                # Prepare tool schemas
                tools_for_binding = []
                for tool in self.tools:
                    tool_dict = {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": {
                                "type": "object",
                                "properties": self._get_tool_properties(tool),
                                "required": self._get_required_fields(tool)
                            }
                        }
                    }
                    tools_for_binding.append(tool_dict)
                
                llm_with_tools = self.llm.bind_tools(tools_for_binding)
                logger.info("✅ Successfully bound %d tools to LLM", len(tools_for_binding))
                return llm_with_tools
            else:
                logger.warning("LLM does not support bind_tools(), will use wrapper mode")
                return None
                
        except Exception as e:
            logger.warning("Failed to bind tools to LLM: %s", e)
            logger.info("Will fall back to wrapper mode with direct tool execution")
            return None
    
    def _get_tool_properties(self, tool: BaseTool) -> Dict[str, Any]:
        """Extract parameter properties from tool's args_schema."""
        try:
            if hasattr(tool, "args_schema") and tool.args_schema:
                schema = tool.args_schema.model_json_schema()
                return schema.get("properties", {})
        except Exception as e:
            logger.warning("Failed to extract properties from %s: %s", tool.name, e)
        
        return {}
    
    def _get_required_fields(self, tool: BaseTool) -> List[str]:
        """Extract required fields from tool's args_schema."""
        try:
            if hasattr(tool, "args_schema") and tool.args_schema:
                schema = tool.args_schema.model_json_schema()
                return schema.get("required", [])
        except Exception as e:
            logger.warning("Failed to extract required fields from %s: %s", tool.name, e)
        
        return []
    
    def invoke(self, user_input: str) -> Dict[str, Any]:
        """
        Execute the agent with the given user input.
        
        Args:
            user_input: The user's input/question
            
        Returns:
            Dictionary with the final output
        """
        logger.info("🚀 Starting manual tool calling execution")
        
        if self.llm_with_tools is not None:
            return self._invoke_with_bound_tools(user_input)
        else:
            return self._invoke_with_wrapper(user_input)
    
    def _invoke_with_bound_tools(self, user_input: str) -> Dict[str, Any]:
        """Execute using bound tools mode."""
        logger.info("📍 Executing with bound tools mode")
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        iteration = 0
        final_response = None
        
        while iteration < self.max_iterations:
            iteration += 1
            logger.debug(f"Iteration {iteration}/{self.max_iterations}")
            
            try:
                # Get response from LLM with tools
                response = self.llm_with_tools.invoke(messages)
                
                # Extract response content and tool calls
                response_content = getattr(response, "content", "")
                tool_calls = getattr(response, "tool_calls", [])
                
                if self.verbose and tool_calls:
                    logger.info(f"🔧 Tool calls detected: {len(tool_calls)}")
                    for call in tool_calls:
                        logger.info(f"   - {call.get('name', 'unknown')} with args: {call.get('args', {})}")
                
                # Process tool calls
                if tool_calls:
                    for tool_call in tool_calls:
                        tool_name = tool_call.get("name")
                        tool_args = tool_call.get("args", {})
                        tool_args = self._sanitize_tool_args(tool_name, tool_args)

                        if self._should_skip_tool_call(tool_name, tool_args):
                            logger.warning("Skipping tool call due to metadata/invalid ticker | tool=%s args=%s", tool_name, tool_args)
                            messages.append({"role": "assistant", "content": response_content})
                            messages.append({
                                "role": "user",
                                "content": f"Tool call skipped for {tool_name}: invalid/metadata ticker input."
                            })
                            continue
                        
                        if tool_name in self.tool_map:
                            logger.info(f"⚙️  Executing tool: {tool_name}")
                            try:
                                short_circuit = self._short_circuit_resolver(tool_name, tool_args)
                                if short_circuit is not None:
                                    tool_result = short_circuit
                                else:
                                    tool_result = self.tool_map[tool_name].run(tool_args)

                                if isinstance(tool_result, dict) and tool_result.get("circuit_breaker"):
                                    message = str(tool_result.get("error", "Circuit breaker triggered"))
                                    logger.error("Circuit breaker received from tool %s: %s", tool_name, message)
                                    return {
                                        "output": message,
                                        "success": False,
                                        "iterations": iteration,
                                        "mode": "bound_tools",
                                        "circuit_breaker": True,
                                    }
                                
                                # Add tool result to messages
                                messages.append({"role": "assistant", "content": response_content})
                                messages.append({
                                    "role": "user",
                                    "content": f"Tool result from {tool_name}: {str(tool_result)[:500]}"
                                })
                                
                                logger.info(f"✅ Tool {tool_name} executed successfully")
                            except Exception as e:
                                logger.error(f"❌ Error executing tool {tool_name}: {e}")
                                messages.append({"role": "assistant", "content": response_content})
                                messages.append({
                                    "role": "user",
                                    "content": f"Tool error in {tool_name}: {str(e)}"
                                })
                        else:
                            logger.warning(f"⚠️  Unknown tool: {tool_name}")
                else:
                    # No tool calls, return final response
                    final_response = response_content
                    logger.info("✅ Final response generated (no tool calls)")
                    break
                    
            except Exception as e:
                logger.error(f"Error during iteration {iteration}: {e}")
                return {
                    "output": f"Error during execution: {str(e)}",
                    "success": False,
                    "iterations": iteration
                }
        
        if final_response is None:
            final_response = "Maximum iterations reached without final response"
        
        return {
            "output": final_response,
            "success": True,
            "iterations": iteration,
            "mode": "bound_tools"
        }

    def _sanitize_tool_args(self, tool_name: Optional[str], tool_args: Any) -> Any:
        """Normalize tool args, especially tickers for BR symbols."""
        if not isinstance(tool_args, dict):
            return tool_args

        normalized_args = dict(tool_args)

        def normalize_symbol(value: Any) -> Any:
            if not isinstance(value, str):
                return value
            cleaned = value.strip().upper()
            if re.match(r"^[A-Z]{4}\d{1,2}$", cleaned):
                return f"{cleaned}.SA"
            return cleaned

        if tool_name in {"finance_report_pull", "rag_fundamental_analysis"}:
            if "symbol" in normalized_args:
                original = normalized_args["symbol"]
                normalized_args["symbol"] = normalize_symbol(original)
                if original != normalized_args["symbol"]:
                    logger.info(
                        "Ticker normalized for %s: %s -> %s",
                        tool_name,
                        original,
                        normalized_args["symbol"],
                    )

        if tool_name == "resolve_company_ticker" and "query" in normalized_args:
            query = normalized_args["query"]
            if isinstance(query, str):
                normalized_args["query"] = query.strip()

        return normalized_args

    def _is_blacklisted_metadata(self, token: str) -> bool:
        return token in METADATA_BLACKLIST

    def _is_valid_ticker_pattern(self, token: str) -> bool:
        return bool(RE_B3_TICKER.match(token) or RE_YAHOO_NYSE_TICKER.match(token))

    def _should_skip_tool_call(self, tool_name: Optional[str], tool_args: Any) -> bool:
        if not isinstance(tool_args, dict):
            return False

        ticker_fields = []
        if tool_name in {"finance_report_pull", "rag_fundamental_analysis"}:
            ticker_fields = ["symbol"]
        elif tool_name == "resolve_company_ticker":
            ticker_fields = ["query"]

        for field in ticker_fields:
            value = tool_args.get(field)
            if not isinstance(value, str):
                continue
            token = value.strip().upper()
            if not token:
                return True
            if self._is_blacklisted_metadata(token):
                return True
            if tool_name in {"finance_report_pull", "rag_fundamental_analysis"} and not self._is_valid_ticker_pattern(token):
                return True

        return False

    def _short_circuit_resolver(self, tool_name: Optional[str], tool_args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Bypass resolve tool for ticker-like inputs to avoid unnecessary refinement loops."""
        if tool_name != "resolve_company_ticker":
            return None

        query = tool_args.get("query")
        if not isinstance(query, str):
            return None

        candidate = query.strip().upper()
        if not candidate:
            return None

        if candidate.endswith(".SA"):
            logger.info("Resolver short-circuit: market ticker detected (%s)", candidate)
            return {
                "success": True,
                "query": query,
                "ticker": candidate,
                "company_name": candidate,
                "resolution_method": "executor_short_circuit_market_ticker",
            }

        if re.match(r"^[A-Z]{4}\d{1,2}$", candidate):
            ticker = f"{candidate}.SA"
            logger.info("Resolver short-circuit: B3 ticker detected (%s -> %s)", candidate, ticker)
            return {
                "success": True,
                "query": query,
                "ticker": ticker,
                "company_name": candidate,
                "resolution_method": "executor_short_circuit_b3_ticker",
            }

        if query == query.strip().upper() and re.match(r"^[A-Z][A-Z0-9\.-]{0,5}$", candidate):
            logger.info("Resolver short-circuit: Yahoo ticker detected (%s)", candidate)
            return {
                "success": True,
                "query": query,
                "ticker": candidate,
                "company_name": candidate,
                "resolution_method": "executor_short_circuit_yahoo_ticker",
            }

        return None
    
    def _invoke_with_wrapper(self, user_input: str) -> Dict[str, Any]:
        """
        Execute using wrapper mode - directly inject tool data in prompt.
        This is the fallback when bind_tools() is not available.
        """
        logger.info("📍 Executing with wrapper mode (tool data injection)")
        
        # Build a rich prompt with tool availability info
        tools_description = "\n".join([
            f"- {tool.name}: {tool.description}" for tool in self.tools
        ])
        
        wrapper_prompt = f"""{self.system_prompt}

AVAILABLE TOOLS (use these to get real data):
{tools_description}

USER REQUEST:
{user_input}

When you need to analyze real financial data, use the available tools to fetch it first, then analyze."""
        
        try:
            # Execute directly with wrapper prompt
            response = self.llm.invoke(wrapper_prompt)
            output = getattr(response, "content", "")
            
            if isinstance(output, list):
                output = "\n".join(str(item) for item in output)
            if not isinstance(output, str):
                output = str(output)
            
            return {
                "output": output,
                "success": True,
                "iterations": 1,
                "mode": "wrapper"
            }
            
        except Exception as e:
            logger.error(f"Error in wrapper mode: {e}")
            return {
                "output": f"Error executing in wrapper mode: {str(e)}",
                "success": False,
                "iterations": 1,
                "mode": "wrapper"
            }
