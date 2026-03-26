"""Fundamental Analysis Agent using Langchain and OpenAI GPT-4o."""

import logging
import os
import re
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import render_text_description
from langchain_openai import ChatOpenAI
from langchain_aws import ChatBedrockConverse
import boto3

try:
    from langchain.agents import create_openai_tools_agent
    from langchain.agents.agent import AgentExecutor
    from langchain.memory import ConversationBufferWindowMemory
    LANGCHAIN_AGENTS_AVAILABLE = True
    LANGCHAIN_AGENTS_IMPORT_ERROR: Optional[Exception] = None
except Exception as import_error:
    create_openai_tools_agent = None  # type: ignore[assignment]
    AgentExecutor = None  # type: ignore[assignment]
    ConversationBufferWindowMemory = None  # type: ignore[assignment]
    LANGCHAIN_AGENTS_AVAILABLE = False
    LANGCHAIN_AGENTS_IMPORT_ERROR = import_error

from .tools import get_fundamental_tools
from .manual_executor import ManualToolCallingExecutor

# Configure LangSmith tracing (opt-in)
langsmith_tracing_enabled = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
if langsmith_tracing_enabled and os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "alpha-agents-fundamental")
else:
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

logger = logging.getLogger(__name__)
METADATA_BLACKLIST = {"ROUND", "AGENT", "TURN", "STEP", "USER", "SYSTEM", "OTHER", "SA"}
RE_B3_TICKER = re.compile(r"^[A-Z]{4}(?:3|4|5|6|11)(?:\.SA)?$")
RE_YAHOO_NYSE_TICKER = re.compile(r"^[A-Z]{1,5}$")


class FundamentalAgent:
    """Langchain-based Fundamental Analysis Agent using OpenAI GPT-4o."""
    
    def __init__(
        self, 
        openai_api_key: Optional[str] = None,
        llm_provider: str = "openai",
        model_name: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4000,
        aws_profile: Optional[str] = None,
        aws_region: str = "us-west-2"
    ):
        """Initialize the Fundamental Agent.
        
        Args:
            openai_api_key: OpenAI API key (if not provided, will use env var)
            llm_provider: LLM provider to use (openai or bedrock)
            model_name: Model name (auto-detected based on provider if None)
            temperature: Model temperature for response generation
            max_tokens: Maximum tokens for model responses
            aws_profile: AWS profile name (for Bedrock)
            aws_region: AWS region (for Bedrock)
        """
        # LLM provider configuration
        self.llm_provider = llm_provider or os.getenv("LLM_PROVIDER", "openai")
        self.aws_profile = aws_profile or os.getenv("AWS_PROFILE")
        self.aws_region = aws_region or os.getenv("AWS_REGION", "us-east-1")
        self.aws_account = os.getenv("AWS_ACCOUNT")  # Will be retrieved if not set
        
        # Set default model based on provider
        if model_name is None:
            if self.llm_provider == "bedrock":
                model_name = os.getenv("LLM_MODEL", "anthropic.claude-sonnet-4-20250514-v1:0")
            else:
                model_name = os.getenv("LLM_MODEL", "gpt-4o")
                
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize tools
        self.tools = get_fundamental_tools()
        
        # Sanitize and inspect tools for Python 3.14 compatibility
        self.tools = self._sanitize_tools(self.tools)
        
        # Initialize LLM based on provider
        if self.llm_provider == "bedrock":
            # Get AWS account ID if not provided
            if not self.aws_account:
                sts_client = boto3.client('sts')
                self.aws_account = sts_client.get_caller_identity()['Account']
            
            # Construct ARN for cross-region inference
            arn_name = (
                f"arn:aws:bedrock:{self.aws_region}:{self.aws_account}:"
                f"inference-profile/us.{model_name}"
            )

            self.llm = ChatBedrockConverse(
                region_name=os.environ['AWS_REGION']
                , model_id=arn_name
                , provider='Anthropic'
                # , model_kwargs=model_kwargs
                # , config={"callbacks": [langfuse_handler]}
            ).with_config(
                tags=["fundamental-agent", "financial-analysis", "sec-filings", "bedrock", "claude"],
                metadata={
                    "agent_name": "fundamental-agent",
                    "agent_type": "specialist",
                    "agent_version": "1.0.0",
                    "system": "alpha-agents",
                    "llm_provider": "bedrock",
                    "model": self.model_name,
                    "capabilities": ["financial_statements", "sec_analysis", "sector_comparison"]
                }
            )
        else:
            # Configuração para DeepSeek (usando interface compatível da OpenAI)
            self.openai_api_key = (openai_api_key or os.getenv("OPENAI_API_KEY", "")).strip()
            if not self.openai_api_key:
                raise ValueError("API key (DeepSeek) is required")
                
            self.llm = ChatOpenAI(
                api_key=self.openai_api_key,
                model="deepseek-chat", # Nome do modelo no DeepSeek
                base_url="https://api.deepseek.com", # O segredo está aqui
                temperature=self.temperature,
                max_tokens=self.max_tokens
            ).with_config(
                tags=["fundamental-agent", "financial-analysis", "sec-filings", "openai"],
                metadata={
                    "agent_name": "fundamental-agent",
                    "agent_type": "specialist",
                    "agent_version": "1.0.0",
                    "system": "alpha-agents",
                    "llm_provider": "openai",
                    "model": self.model_name,
                    "capabilities": ["financial_statements", "sec_analysis", "sector_comparison"]
                }
            )
        
        # Create system prompt
        self.system_prompt = self._create_system_prompt()

        self.agent_mode_enabled = LANGCHAIN_AGENTS_AVAILABLE
        self.conversation_history: List[Dict[str, Any]] = []
        
        # Initialize agent-related attributes (ALWAYS initialized to prevent AttributeError)
        self.memory = None
        self.agent = None
        self.agent_executor = None
        
        # Try to create agent executor using manual tool calling (Python 3.14 compatible)
        try:
            logger.info("Attempting to create manual tool calling executor...")
            self.agent_executor = ManualToolCallingExecutor(
                llm=self.llm,
                tools=self.tools,
                system_prompt=self.system_prompt,
                max_iterations=10,
                verbose=True
            )
            self.agent_mode_enabled = True
            logger.info("✅ Manual tool calling executor created successfully")
        except Exception as e:
            logger.error(
                "Failed to create manual tool calling executor (%s: %s)",
                type(e).__name__, e, exc_info=True
            )
            self.agent_executor = None
            self.agent_mode_enabled = False
        
        logger.info(
            "Fundamental Agent initialized successfully (mode: %s)",
            "AGENT (manual tool calling)" if self.agent_executor else "DIRECT_LLM"
        )

    def _create_system_prompt(self) -> str:
        """Create the system prompt for the fundamental agent."""
        return """You are a specialized Fundamental Analysis Agent for equity research and portfolio management.

Your core expertise includes:

**Financial Statement Analysis:**
- Deep analysis of income statements, balance sheets, and cash flow statements
- Assessment of financial health, profitability, and operational efficiency
- Identification of trends, patterns, and anomalies in financial data
- Evaluation of working capital management and capital allocation decisions

**SEC Filing Analysis:**  
- Comprehensive review of 10-K and 10-Q reports
- Management Discussion & Analysis (MD&A) insights
- Identification of business risks and opportunities
- Assessment of management quality and strategic direction

**Fundamental Valuation:**
- DCF modeling and intrinsic value estimation
- Multiple-based valuation (P/E, P/B, EV/EBITDA, PEG)
- Sum-of-the-parts analysis for complex businesses
- Sensitivity analysis and scenario modeling

**Sector and Industry Analysis:**
- Industry comparison and competitive benchmarking  
- Sector trend identification and market cycle analysis
- Regulatory and macroeconomic impact assessment
- ESG considerations and sustainability analysis

**Investment Decision Framework:**
Your analysis should result in clear BUY/SELL/HOLD recommendations with:
- Target price estimates with supporting rationale
- Risk assessment (financial, operational, market risks)
- Time horizon considerations (short-term vs. long-term outlook)
- Catalyst identification (upcoming events, product launches, earnings)

**Risk Tolerance Considerations:**
- Risk-averse: Focus on stable, dividend-paying stocks with strong balance sheets
- Risk-neutral: Balance growth potential with financial stability
- Risk-seeking: Consider high-growth opportunities with acceptable fundamental risks

**Analysis Methodology:**
1. Use the finance_report_pull tool to gather comprehensive financial data
2. Apply the rag_fundamental_analysis tool for in-depth analysis of specific areas
3. Integrate quantitative metrics with qualitative business assessment
4. Provide evidence-based recommendations with clear reasoning

**Available Tools:**
- resolve_company_ticker: Convert company names to stock tickers
- finance_report_pull: Retrieve comprehensive financial reports with data validation
- rag_fundamental_analysis: Perform detailed analysis of cash flow, operations, risks, and strategic progress

Always provide thorough, evidence-based analysis with specific financial metrics and ratios to support your conclusions."""

    def _sanitize_tools(self, tools: List[Any]) -> List[Any]:
        """Sanitize tools for Python 3.14 compatibility by removing problematic type annotations."""
        sanitized = []
        for tool in tools:
            try:
                # Log tool inspection
                logger.debug(f"Inspecting tool: {tool.name}")
                
                # Check for __annotations__ that might cause issues in Python 3.14
                if hasattr(tool, "__annotations__") and tool.__annotations__:
                    logger.debug(f"Tool {tool.name} has annotations: {list(tool.__annotations__.keys())}")
                
                # Check args_schema
                if hasattr(tool, "args_schema") and tool.args_schema:
                    logger.debug(f"Tool {tool.name} args_schema: {tool.args_schema.__name__}")
                
                sanitized.append(tool)
            except Exception as e:
                logger.warning(f"Error inspecting tool {getattr(tool, 'name', 'unknown')}: {e}")
                sanitized.append(tool)
        
        logger.info(f"Sanitized {len(sanitized)} tools for compatibility")
        return sanitized

    def _is_blacklisted_metadata(self, value: str) -> bool:
        return value.strip().upper() in METADATA_BLACKLIST

    def _looks_like_valid_ticker(self, value: str) -> bool:
        token = value.strip().upper()
        return bool(RE_B3_TICKER.match(token) or RE_YAHOO_NYSE_TICKER.match(token))

    def _run_prompt(self, prompt_text: str) -> str:
        """Run prompt through agent executor when available, else direct LLM fallback.
        
        Args:
            prompt_text: The prompt to execute

        Returns:
            Response text from either agent executor or direct LLM
        """
        try:
            # Use agent executor (manual tool calling) if available
            if self.agent_executor is not None:
                logger.debug("Using manual tool calling executor")
                result = self.agent_executor.invoke(prompt_text)
                output_text = result.get("output", "")

                if result.get("circuit_breaker"):
                    raise RuntimeError(output_text or "Circuit breaker triggered in fundamental tool execution")
                
                # Log the mode used
                mode = result.get("mode", "unknown")
                iterations = result.get("iterations", 0)
                logger.info(f"✅ Executed in {mode} mode ({iterations} iteration(s))")
                
                # Ensure output is string
                if not isinstance(output_text, str):
                    output_text = str(output_text)
                
                self.conversation_history.append({"role": "human", "content": prompt_text})
                self.conversation_history.append({"role": "assistant", "content": output_text})
                return output_text
            
            # Fallback to direct LLM if no executor available
            logger.debug("⚠️ No executor available - using direct LLM fallback")
            combined_prompt = (
                f"{self.system_prompt}\n\n"
                f"User request:\n{prompt_text}\n\n"
                "Respond with a complete, structured fundamental analysis."
            )
            
            response = self.llm.invoke(combined_prompt)
            output_text = getattr(response, "content", "")
            
            if isinstance(output_text, list):
                output_text = "\n".join(str(item) for item in output_text)
            if not isinstance(output_text, str):
                output_text = str(output_text)

            self.conversation_history.append({"role": "human", "content": prompt_text})
            self.conversation_history.append({"role": "assistant", "content": output_text})
            return output_text
        
        except Exception as e:
            logger.error(f"Error in _run_prompt: {type(e).__name__}: {e}", exc_info=True)
            raise

    def analyze_fundamental(
        self,
        stock_input: str,
        risk_tolerance: str = "neutral",
        analysis_depth: str = "comprehensive",
        focus_areas: Optional[List[str]] = None,
        context: str = ""
    ) -> Dict[str, Any]:
        """Perform comprehensive fundamental analysis.
        
        Args:
            stock_input: Stock ticker, company name, or ISIN
            risk_tolerance: Investment risk profile ("averse", "neutral", "seeking")
            analysis_depth: Level of analysis ("quick", "standard", "comprehensive") 
            focus_areas: Specific areas to focus on (e.g., ["cash_flow", "operations"])
            context: Additional context or specific questions
            
        Returns:
            Dictionary containing fundamental analysis results
        """
        try:
            stock_token = (stock_input or "").strip().upper()
            if self._is_blacklisted_metadata(stock_token):
                return {
                    "success": False,
                    "stock_input": stock_input,
                    "error": f"Blocked metadata token as stock input: {stock_token}"
                }

            # Prepare the analysis prompt
            analysis_prompt = f"""Please perform comprehensive fundamental analysis for: {stock_input}

Analysis Requirements:
- Risk Tolerance: {risk_tolerance}
- Analysis Depth: {analysis_depth}
- Focus Areas: {', '.join(focus_areas) if focus_areas else 'All key areas'}

{f"Additional Context: {context}" if context else ""}

Please provide:
1. Company overview and business model assessment
2. Financial health analysis (balance sheet strength, liquidity, leverage)
3. Profitability analysis (margins, ROE, ROA trends)
4. Cash flow quality and sustainability assessment  
5. Growth prospects and competitive position
6. Valuation analysis with target price estimate
7. Key risks and potential catalysts
8. Clear BUY/SELL/HOLD recommendation with rationale

Use the available tools to gather financial data and perform detailed analysis. Present findings in a structured, professional format suitable for investment decision-making."""

            output_text = self._run_prompt(analysis_prompt)
            
            return {
                "success": True,
                "stock_input": stock_input,
                "analysis": output_text,
                "risk_tolerance": risk_tolerance,
                "analysis_depth": analysis_depth,
                "tools_used": [tool.name for tool in self.tools]
            }
            
        except Exception as e:
            logger.error(f"Error in fundamental analysis for {stock_input}: {str(e)}")
            return {
                "success": False,
                "stock_input": stock_input,
                "error": f"Fundamental analysis failed: {str(e)}"
            }

    def quick_valuation(
        self,
        ticker: str,
        valuation_method: str = "dcf",
        risk_tolerance: str = "neutral"
    ) -> Dict[str, Any]:
        """Perform a quick valuation analysis.
        
        Args:
            ticker: Stock ticker symbol
            valuation_method: Valuation approach ("dcf", "multiples", "hybrid")
            risk_tolerance: Risk profile for assumptions
            
        Returns:
            Dictionary with quick valuation assessment
        """
        try:
            ticker_token = (ticker or "").strip().upper()
            if self._is_blacklisted_metadata(ticker_token):
                return {
                    "success": False,
                    "ticker": ticker,
                    "error": f"Blocked metadata token as ticker input: {ticker_token}"
                }

            valuation_prompt = f"""Perform quick fundamental valuation for {ticker}:

Valuation Method: {valuation_method}
Risk Profile: {risk_tolerance}

Please provide:
1. Current financial metrics and key ratios
2. {valuation_method.upper()} valuation with target price
3. Comparison to current market price
4. Key valuation drivers and assumptions
5. Quick BUY/SELL/HOLD recommendation

Keep analysis concise but substantive."""

            output_text = self._run_prompt(valuation_prompt)
            
            return {
                "success": True,
                "ticker": ticker,
                "valuation_method": valuation_method,
                "analysis": output_text,
                "risk_tolerance": risk_tolerance
            }
            
        except Exception as e:
            logger.error(f"Error in quick valuation for {ticker}: {str(e)}")
            return {
                "success": False,
                "ticker": ticker,
                "error": f"Quick valuation failed: {str(e)}"
            }

    def sector_comparison(
        self,
        tickers: List[str],
        analysis_focus: str = "fundamental_metrics",
        risk_tolerance: str = "neutral"
    ) -> Dict[str, Any]:
        """Compare fundamental metrics across multiple stocks.
        
        Args:
            tickers: List of stock tickers to compare
            analysis_focus: Focus area for comparison
            risk_tolerance: Risk profile for evaluation
            
        Returns:
            Dictionary with sector comparison results
        """
        try:
            if len(tickers) > 5:
                return {
                    "success": False,
                    "error": "Maximum 5 stocks supported for comparison"
                }
            
            comparison_prompt = f"""Compare fundamental metrics for these stocks: {', '.join(tickers)}

Analysis Focus: {analysis_focus}
Risk Profile: {risk_tolerance}

Please provide:
1. Key fundamental metrics comparison table
2. Relative valuation analysis
3. Competitive positioning assessment
4. Risk-adjusted investment rankings
5. Top pick recommendation with rationale

Focus on quantitative comparisons with clear reasoning."""

            output_text = self._run_prompt(comparison_prompt)
            
            return {
                "success": True,
                "tickers": tickers,
                "analysis_focus": analysis_focus,
                "comparison": output_text,
                "risk_tolerance": risk_tolerance
            }
            
        except Exception as e:
            logger.error(f"Error in sector comparison: {str(e)}")
            return {
                "success": False,
                "tickers": tickers,
                "error": f"Sector comparison failed: {str(e)}"
            }

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history."""
        try:
            if self.memory is not None:
                messages = self.memory.chat_memory.messages
                return [
                    {
                        "role": msg.type,
                        "content": msg.content,
                        "timestamp": getattr(msg, 'timestamp', None)
                    }
                    for msg in messages
                ]
            return self.conversation_history[-20:]
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {str(e)}")
            return []

    def clear_conversation_history(self):
        """Clear the conversation history."""
        try:
            if self.memory is not None:
                self.memory.clear()
            self.conversation_history.clear()
            logger.info("Conversation history cleared")
        except Exception as e:
            logger.error(f"Error clearing conversation history: {str(e)}")