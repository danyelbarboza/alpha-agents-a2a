"""Tools for the Sentiment Agent - News collection and sentiment analysis."""

import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

import dateutil.parser
import feedparser
import httpx
import yfinance as yf
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)


class NewsCollectionInput(BaseModel):
    """Input schema for news collection."""

    symbol: str = Field(description="Stock symbol (e.g., AAPL, TSLA, MSFT)")
    max_articles: int = Field(
        default=10,
        description="Maximum number of news articles to collect"
    )
    lookback_days: int = Field(
        default=7,
        description="Number of days to look back for news"
    )


class SentimentAnalysisInput(BaseModel):
    """Input schema for sentiment analysis."""

    symbol: str = Field(description="Stock symbol (e.g., AAPL, TSLA, MSFT)")
    news_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Pre-collected news data (if not provided, will collect news first)"
    )


class CompanyResolverInput(BaseModel):
    """Input schema for company name resolution."""

    query: str = Field(description="Company name, ISIN, or potential ticker symbol to resolve")


class StockNewsCollectionTool(BaseTool):
    """Tool to collect stock-related news from multiple sources."""

    name: str = "collect_stock_news"
    description: str = (
        "Collects recent financial news related to a specific stock from multiple sources "
        "including Yahoo Finance, Google News, and financial RSS feeds. Returns structured "
        "news data with headlines, sources, publication dates, and content."
    )
    args_schema: type[BaseModel] = NewsCollectionInput

    async def _get_session(self) -> httpx.AsyncClient:
        """Get or create HTTP session."""
        user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        )
        return httpx.AsyncClient(
            timeout=30.0,
            headers={
                "User-Agent": user_agent,
                "Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8",
                "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
                "Cache-Control": "no-cache",
            }
        )

    def _run(self, symbol: str, max_articles: int = 10, lookback_days: int = 7) -> Dict[str, Any]:
        """Collect stock news synchronously."""
        return asyncio.run(self._arun(symbol, max_articles, lookback_days))

    async def _arun(self, symbol: str, max_articles: int = 10, lookback_days: int = 7) -> Dict[str, Any]:
        """Collect stock news asynchronously."""
        session = None
        try:
            # Get company information first
            normalized_symbol = self._normalize_symbol(symbol)
            ticker = yf.Ticker(normalized_symbol)
            try:
                info = ticker.info
                company_name = info.get('longName', normalized_symbol)
            except:
                company_name = normalized_symbol

            logger.info(f"Collecting news for {normalized_symbol} ({company_name})")

            # Create session for news collection
            session = await self._get_session()

            # Collect news from multiple sources
            all_articles: List[Dict[str, Any]] = []
            search_terms = self._build_search_terms(normalized_symbol, company_name)
            logger.info("News search terms for %s: %s", normalized_symbol, search_terms)

            per_source_limit = max(3, max_articles // 2)
            for idx, term in enumerate(search_terms):
                term_articles = await self._collect_news_with_retry(
                    session=session,
                    symbol=normalized_symbol,
                    company_name=company_name,
                    search_term=term,
                    max_articles=per_source_limit,
                    prefer_google_first=(idx % 2 == 1),
                )
                all_articles.extend(term_articles)

                if len(all_articles) >= max_articles:
                    break

            # Filter by date and deduplicate
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=lookback_days)
            filtered_articles = []
            seen_titles = set()

            for article in all_articles:
                # Skip duplicates based on title similarity
                title_key = self._normalize_title(article.get('title', ''))
                if title_key in seen_titles:
                    continue
                seen_titles.add(title_key)

                # Check date
                pub_date = article.get('published_date')
                if pub_date:
                    # Ensure both dates are timezone-aware for comparison
                    if pub_date.tzinfo is None:
                        pub_date = pub_date.replace(tzinfo=timezone.utc)
                    if pub_date >= cutoff_date:
                        filtered_articles.append(article)
                else:
                    # Include articles without dates
                    filtered_articles.append(article)

            # Sort by date (newest first) and limit
            filtered_articles.sort(key=lambda x: x.get('published_date', datetime.min), reverse=True)
            final_articles = filtered_articles[:max_articles]

            if not final_articles:
                logger.warning(
                    "No news articles collected for symbol=%s after retries and fallback terms",
                    normalized_symbol,
                )

            result = {
                "success": True,
                "symbol": normalized_symbol,
                "company_name": company_name,
                "collection_date": datetime.now(timezone.utc).isoformat(),
                "lookback_days": lookback_days,
                "articles_collected": len(final_articles),
                "search_terms_used": search_terms,
                "articles": final_articles
            }

            logger.info(f"Successfully collected {len(final_articles)} articles for {normalized_symbol}")
            return result

        except Exception as e:
            logger.error(f"Error collecting news for {symbol}: {str(e)}")
            return {
                "success": False,
                "symbol": symbol,
                "error": f"Failed to collect news: {str(e)}"
            }

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize ticker symbol, adding .SA for BR-like tickers when needed."""
        cleaned = (symbol or "").strip().upper()
        if re.match(r"^[A-Z]{4}\d{1,2}$", cleaned):
            return f"{cleaned}.SA"
        return cleaned

    def _build_search_terms(self, symbol: str, company_name: str) -> List[str]:
        """Build fallback search terms for RSS sources."""
        terms: List[str] = []

        def add_term(value: str):
            value = (value or "").strip()
            if value and value not in terms:
                terms.append(value)

        normalized_symbol = self._normalize_symbol(symbol)
        base_symbol = normalized_symbol.replace(".SA", "")

        add_term(normalized_symbol)
        add_term(base_symbol)
        add_term(f"{company_name} notícias financeiras")
        add_term(f"{base_symbol} B3")

        return terms

    async def _collect_news_with_retry(
        self,
        session: httpx.AsyncClient,
        symbol: str,
        company_name: str,
        search_term: str,
        max_articles: int,
        prefer_google_first: bool = False,
    ) -> List[Dict[str, Any]]:
        """Collect from Yahoo/Google with exponential backoff and source failover."""
        max_attempts = 3
        source_order = ["google", "yahoo"] if prefer_google_first else ["yahoo", "google"]
        collected: List[Dict[str, Any]] = []

        for attempt in range(1, max_attempts + 1):
            logger.info(
                "RSS attempt %s/%s | symbol=%s | term=%s | source_order=%s",
                attempt,
                max_attempts,
                symbol,
                search_term,
                source_order,
            )

            for source_name in source_order:
                try:
                    if source_name == "yahoo":
                        yahoo_articles = await self._collect_yahoo_finance_news(
                            session=session,
                            symbol=self._symbol_for_yahoo(search_term, symbol),
                            max_articles=max_articles,
                        )
                        collected.extend(yahoo_articles)
                    else:
                        google_articles = await self._collect_google_news(
                            session=session,
                            symbol=symbol,
                            company_name=company_name,
                            max_articles=max_articles,
                            search_term=search_term,
                        )
                        collected.extend(google_articles)
                except Exception as e:
                    logger.warning(
                        "RSS source error | source=%s | term=%s | error=%s: %s",
                        source_name,
                        search_term,
                        type(e).__name__,
                        e,
                    )

            if collected:
                logger.info(
                    "RSS collected %s articles on attempt %s for term=%s",
                    len(collected),
                    attempt,
                    search_term,
                )
                return collected

            if attempt < max_attempts:
                wait_seconds = 2 ** attempt
                logger.info(
                    "RSS returned no items for term=%s on attempt %s; retrying in %ss",
                    search_term,
                    attempt,
                    wait_seconds,
                )
                await asyncio.sleep(wait_seconds)

        logger.warning(
            "RSS exhausted retries with zero items | symbol=%s | term=%s | reason=empty_or_connection_error",
            symbol,
            search_term,
        )
        return collected

    def _symbol_for_yahoo(self, search_term: str, fallback_symbol: str) -> str:
        """Choose the best Yahoo symbol query from a search term."""
        normalized_fallback = self._normalize_symbol(fallback_symbol)
        candidate = (search_term or "").strip().upper()
        if re.match(r"^[A-Z]{4}\d{1,2}(\.SA)?$", candidate):
            return self._normalize_symbol(candidate)
        return normalized_fallback

    async def _collect_yahoo_finance_news(self, session: httpx.AsyncClient, symbol: str, max_articles: int) -> List[Dict[str, Any]]:
        """Collect news from Yahoo Finance."""
        articles = []
        try:
            # Yahoo Finance RSS feed
            rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={quote_plus(symbol)}&region=BR&lang=pt-BR"
            logger.info("Yahoo RSS URL: %s", rss_url)

            response = await session.get(rss_url)
            response.raise_for_status()

            feed = feedparser.parse(response.text)
            if getattr(feed, "bozo", 0):
                logger.warning("Yahoo RSS parse warning for %s: %s", symbol, getattr(feed, "bozo_exception", None))

            logger.info("Yahoo RSS items for %s: %s", symbol, len(feed.entries))

            for entry in feed.entries[:max_articles]:
                article = {
                    "title": entry.get('title', ''),
                    "url": entry.get('link', ''),
                    "source": "Yahoo Finance",
                    "published_date": self._parse_date(entry.get('published', '')),
                    "summary": entry.get('summary', ''),
                    "content": entry.get('summary', '')  # RSS usually only has summary
                }
                articles.append(article)

            if not articles:
                logger.warning("Yahoo RSS returned empty list for symbol=%s | reason=empty_feed", symbol)

        except Exception as e:
            status_code = getattr(getattr(e, "response", None), "status_code", None)
            logger.warning(
                "Failed to collect Yahoo Finance news | symbol=%s | http_status=%s | error=%s: %s",
                symbol,
                status_code,
                type(e).__name__,
                str(e),
            )

        return articles

    async def _collect_google_news(
        self,
        session: httpx.AsyncClient,
        symbol: str,
        company_name: str,
        max_articles: int,
        search_term: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Collect news from Google News RSS."""
        articles = []
        try:
            # Google News RSS search
            query = search_term or f'"{company_name}" OR "{symbol}" notícias financeiras'
            encoded_query = quote_plus(query)
            rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=pt-BR&gl=BR&ceid=BR:pt-419"
            logger.info("Google RSS URL: %s", rss_url)

            response = await session.get(rss_url)
            response.raise_for_status()

            feed = feedparser.parse(response.text)
            if getattr(feed, "bozo", 0):
                logger.warning("Google RSS parse warning for term '%s': %s", query, getattr(feed, "bozo_exception", None))

            logger.info("Google RSS items for term '%s': %s", query, len(feed.entries))

            for entry in feed.entries[:max_articles]:
                article = {
                    "title": entry.get('title', ''),
                    "url": entry.get('link', ''),
                    "source": self._extract_source_from_google_news(entry),
                    "published_date": self._parse_date(entry.get('published', '')),
                    "summary": entry.get('title', ''),  # Google News RSS doesn't have summary
                    "content": entry.get('title', '')
                }
                articles.append(article)

            if not articles:
                logger.warning("Google RSS returned empty list for term='%s' | reason=empty_feed", query)

        except Exception as e:
            status_code = getattr(getattr(e, "response", None), "status_code", None)
            logger.warning(
                "Failed to collect Google News | term='%s' | http_status=%s | error=%s: %s",
                search_term or symbol,
                status_code,
                type(e).__name__,
                str(e),
            )

        return articles

    def _extract_source_from_google_news(self, entry) -> str:
        """Extract source from Google News entry."""
        try:
            # Google News includes source in the title usually
            title = entry.get('title', '')
            if ' - ' in title:
                return title.split(' - ')[-1]
            return "Google News"
        except:
            return "Google News"

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime object."""
        if not date_str:
            return None

        try:
            # Try parsing common date formats
            parsed_date = dateutil.parser.parse(date_str)
            # Ensure timezone-aware datetime
            if parsed_date.tzinfo is None:
                parsed_date = parsed_date.replace(tzinfo=timezone.utc)
            return parsed_date
        except:
            return None

    def _normalize_title(self, title: str) -> str:
        """Normalize title for deduplication."""
        # Remove special characters, convert to lowercase, remove extra spaces
        normalized = re.sub(r'[^\w\s]', '', title.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized[:50]  # Take first 50 chars for comparison


class CompanyNameResolverTool(BaseTool):
    """Tool to resolve company names or ISIN to stock tickers."""

    name: str = "resolve_company_ticker"
    description: str = (
        "Resolves company names (e.g., 'Apple', 'Microsoft') or ISIN codes to stock tickers. "
        "Returns the ticker symbol that can be used for news collection."
    )
    args_schema: type[BaseModel] = CompanyResolverInput

    def _run(self, query: str) -> Dict[str, Any]:
        """Resolve company name/ISIN to ticker symbol."""
        try:
            query = query.strip()

            # Common company name mappings for financial news
            company_mappings = {
                # --- Principais Brasileiras (B3) ---
                "petrobras": "PETR4.SA",
                "petrobrás": "PETR4.SA",
                "vale": "VALE3.SA",
                "itau": "ITUB4.SA",
                "itau unibanco": "ITUB4.SA",
                "bradesco": "BBDC4.SA",
                "banco do brasil": "BBAS3.SA",
                "santander brasil": "SANB11.SA",
                "btg pactual": "BPAC11.SA",
                "b3": "B3SA3.SA",
                "ambev": "ABEV3.SA",
                "weg": "WEGE3.SA",
                "magazine luiza": "MGLU3.SA",
                "magalu": "MGLU3.SA",
                "americanas": "AMER3.SA",
                "via": "BHIA3.SA",
                "casas bahia": "BHIA3.SA",
                "lojas renner": "LREN3.SA",
                "natura": "NTCO3.SA",
                "jbs": "JBSS3.SA",
                "brf": "BRFS3.SA",
                "marfrig": "MRFG3.SA",
                "suzano": "SUZB3.SA",
                "klabin": "KLBN11.SA",
                "gerdau": "GGBR4.SA",
                "csn": "CSNA3.SA",
                "usiminas": "USIM5.SA",
                "embraer": "EMBR3.SA",
                "localiza": "RENT3.SA",
                "azul": "AZUL4.SA",
                "gol": "GOLL4.SA",
                "eletrobras": "ELET3.SA",
                "eletrobrás": "ELET3.SA",
                "petro rio": "PRIO3.SA",
                "prio": "PRIO3.SA",
                "ultrapar": "UGPA3.SA",
                "vibra energia": "VBBR3.SA",
                "raia drogasil": "RADL3.SA",
                "drogasil": "RADL3.SA",
                "hypera": "HYPE3.SA",
                "equatorial": "EQTL3.SA",
                "energisa": "ENGI11.SA",
                "cpfl": "CPFE3.SA",
                "sabesp": "SBSP3.SA",
                "ccr": "CCRO3.SA",
                "ecoRodovias": "ECOR3.SA",
                "mrv": "MRVE3.SA",
                "cyrela": "CYRE3.SA",
                "multiplan": "MULT3.SA",
                "iguatemi": "IGTI11.SA",
                "totvs": "TOTS3.SA",
                "tim brasil": "TIMS3.SA",
                "vivo": "VIVT3.SA",
                "telefônica brasil": "VIVT3.SA",
                "raízen": "RAIZ4.SA",
                "raizen": "RAIZ4.SA",
                "arezzo": "ARZZ3.SA",
                "haring": "HGTX3.SA",

                # --- Internacionais (Mantidas) ---
                "apple": "AAPL",
                "microsoft": "MSFT",
                "google": "GOOGL",
                "alphabet": "GOOGL",
                "amazon": "AMZN",
                "tesla": "TSLA",
                "meta": "META",
                "nvidia": "NVDA",
                "jpmorgan": "JPM",
                "walmart": "WMT",
                "disney": "DIS",
                "coca cola": "KO",
                "netflix": "NFLX"
            }
            # Check if it's already a ticker
            if query.isupper() and 1 <= len(query) <= 5 and query.isalpha():
                try:
                    ticker = yf.Ticker(query)
                    info = ticker.info
                    if info and 'symbol' in info:
                        return {
                            "success": True,
                            "query": query,
                            "ticker": query,
                            "company_name": info.get('longName', 'Unknown'),
                            "resolution_method": "direct_ticker"
                        }
                except:
                    pass

            # Check company mappings
            query_lower = query.lower()
            if query_lower in company_mappings:
                ticker = company_mappings[query_lower]
                try:
                    yf_ticker = yf.Ticker(ticker)
                    info = yf_ticker.info
                    return {
                        "success": True,
                        "query": query,
                        "ticker": ticker,
                        "company_name": info.get('longName', 'Unknown'),
                        "resolution_method": "company_mapping"
                    }
                except Exception as e:
                    logger.warning(f"Validation failed for mapped ticker {ticker}: {e}")

            return {
                "success": False,
                "query": query,
                "error": f"Could not resolve '{query}' to a valid stock ticker",
                "suggestions": [
                    "Try using the stock ticker directly (e.g., AAPL for Apple)",
                    "Check spelling of company name",
                    "Use well-known company names like 'Apple', 'Microsoft', 'Google', etc."
                ]
            }

        except Exception as e:
            logger.error(f"Error resolving ticker for {query}: {str(e)}")
            return {
                "success": False,
                "query": query,
                "error": f"Failed to resolve ticker: {str(e)}"
            }

    async def _arun(self, query: str) -> Dict[str, Any]:
        """Resolve company name/ISIN to ticker symbol asynchronously."""
        return await asyncio.to_thread(self._run, query)


# Export all tools
def get_sentiment_tools() -> List[BaseTool]:
    """Get all sentiment analysis tools."""
    return [
        CompanyNameResolverTool(),
        StockNewsCollectionTool()
    ]
