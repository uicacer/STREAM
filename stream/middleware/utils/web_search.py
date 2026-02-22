"""
Web Search — Internet connectivity for LLM queries.

This module provides web search and URL fetching capabilities that can be
used to augment LLM responses with current information from the internet.

ARCHITECTURE:
  When the user enables web search, this module is called BEFORE the LLM
  receives the conversation. It works in three steps:

  1. SEARCH: Query a search engine (DuckDuckGo or Tavily) for relevant results
  2. FETCH:  If the user's message contains URLs, fetch their content
  3. FORMAT: Combine search results and fetched content into a system message
             that gets prepended to the conversation

  The LLM then sees the search results as part of its context and can
  reference them in its response. This approach works with ALL models
  across ALL tiers because it doesn't require tool calling support.

WHY PRE-QUERY INJECTION (NOT TOOL CALLING):
  Tool calling requires the model to:
    1. Recognize it needs to search
    2. Emit a tool_call response
    3. Wait for the tool result
    4. Generate a final response
  This multi-turn dance requires specific model support (not all models
  have it, especially small Ollama models like Llama 3.2 3B). By injecting
  search results BEFORE the LLM call, we make it work with every model.

PROVIDERS:
  - DuckDuckGo (default): Free, no API key needed. Uses the duckduckgo-search
    library which scrapes DuckDuckGo results. Good for desktop mode where
    users don't want to manage API keys.

  - Tavily (optional): AI-optimized search API. Returns higher quality,
    pre-extracted content designed for LLM consumption. Requires an API
    key ($0 for 1000 searches/month free tier). Better for server deployments.

URL FETCHING:
  When the user pastes a URL in their message (e.g., "summarize this:
  https://example.com/article"), we detect it with a regex, fetch the
  page content, extract readable text with BeautifulSoup, and include
  it in the context. This works independently of the search provider.

USAGE:
  from stream.middleware.utils.web_search import perform_web_search

  # Returns a formatted system message string (or None if search fails)
  context = await perform_web_search(
      query="latest Python 3.13 features",
      provider="duckduckgo",
      tavily_api_key=None,
  )
"""

import logging
import re

import httpx
from bs4 import BeautifulSoup

from stream.middleware.config import (
    WEB_SEARCH_MAX_CONTENT_LENGTH,
    WEB_SEARCH_MAX_RESULTS,
    WEB_SEARCH_TIMEOUT,
)

logger = logging.getLogger(__name__)

# Regex to detect URLs in user messages.
# Matches http:// and https:// URLs. We intentionally keep this simple —
# complex URL regex patterns have diminishing returns and can match things
# that aren't URLs. The user can always paste a clean URL.
URL_PATTERN = re.compile(
    r"https?://[^\s<>\"')\]]+",
    re.IGNORECASE,
)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


class SearchResult:
    """A single search result from any provider."""

    def __init__(self, title: str, url: str, content: str):
        self.title = title
        self.url = url
        self.content = content

    def __repr__(self) -> str:
        return f"SearchResult(title={self.title!r}, url={self.url!r})"


# =============================================================================
# URL DETECTION
# =============================================================================


def extract_urls(text: str) -> list[str]:
    """Extract all URLs from a text string.

    Returns a deduplicated list of URLs found in the text, preserving
    the order of first appearance.

    Examples:
        >>> extract_urls("Check out https://example.com and https://other.org")
        ['https://example.com', 'https://other.org']

        >>> extract_urls("No URLs here")
        []
    """
    seen: set[str] = set()
    urls: list[str] = []
    for match in URL_PATTERN.finditer(text):
        url = match.group(0)
        # Strip trailing punctuation that might have been captured
        url = url.rstrip(".,;:!?)")
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


# =============================================================================
# URL CONTENT FETCHING
# =============================================================================


async def fetch_url_content(url: str) -> str | None:
    """Fetch a URL and extract its readable text content.

    Uses httpx for async HTTP requests and BeautifulSoup for HTML parsing.
    Returns the extracted text truncated to WEB_SEARCH_MAX_CONTENT_LENGTH,
    or None if the fetch fails.

    WHY BeautifulSoup:
      Raw HTML is full of tags, scripts, styles, and navigation elements
      that waste LLM context tokens. BeautifulSoup extracts just the
      readable text, which is what the LLM needs to understand the page.

    Args:
        url: The URL to fetch.

    Returns:
        Extracted text content (truncated), or None on failure.
    """
    try:
        async with httpx.AsyncClient(
            timeout=WEB_SEARCH_TIMEOUT,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; STREAM-AI/1.0; " "+https://github.com/uicacer/STREAM)"
                ),
            },
        ) as client:
            response = await client.get(url)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type and "text/plain" not in content_type:
                logger.debug(f"[WebSearch] Skipping non-text content: {content_type} for {url}")
                return None

            if "text/plain" in content_type:
                text = response.text[:WEB_SEARCH_MAX_CONTENT_LENGTH]
                return text.strip() if text.strip() else None

            # Parse HTML and extract readable text
            soup = BeautifulSoup(response.text, "lxml")

            # Remove elements that don't contain useful content
            for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)

            # Collapse multiple blank lines into single newlines
            text = re.sub(r"\n{3,}", "\n\n", text)

            if not text.strip():
                return None

            return text[:WEB_SEARCH_MAX_CONTENT_LENGTH]

    except httpx.HTTPStatusError as e:
        logger.warning(f"[WebSearch] HTTP error fetching {url}: {e.response.status_code}")
        return None
    except httpx.TimeoutException:
        logger.warning(f"[WebSearch] Timeout fetching {url}")
        return None
    except Exception as e:
        logger.warning(f"[WebSearch] Error fetching {url}: {e}")
        return None


# =============================================================================
# DUCKDUCKGO SEARCH
# =============================================================================


async def _search_duckduckgo(query: str) -> list[SearchResult]:
    """Search the web using DuckDuckGo (free, no API key needed).

    Uses the duckduckgo-search library which scrapes DuckDuckGo's results.
    This is NOT an official API — DuckDuckGo could change their format at
    any time. However, the library is actively maintained (v8.0+) and
    handles retries/fallbacks internally.

    RATE LIMITING:
      DuckDuckGo may throttle or block requests from a single IP if too
      many are sent. For single-user desktop mode this is rarely an issue.
      For multi-user server deployments, consider using Tavily instead.

    Args:
        query: The search query string.

    Returns:
        List of SearchResult objects (up to WEB_SEARCH_MAX_RESULTS).
    """
    try:
        from duckduckgo_search import DDGS

        results: list[SearchResult] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=WEB_SEARCH_MAX_RESULTS):
                results.append(
                    SearchResult(
                        title=r.get("title", ""),
                        url=r.get("href", ""),
                        content=r.get("body", ""),
                    )
                )
        return results

    except ImportError:
        logger.error(
            "[WebSearch] duckduckgo-search not installed. Run: pip install duckduckgo-search"
        )
        return []
    except Exception as e:
        logger.warning(f"[WebSearch] DuckDuckGo search failed: {e}")
        return []


# =============================================================================
# TAVILY SEARCH
# =============================================================================


async def _search_tavily(query: str, api_key: str) -> list[SearchResult]:
    """Search the web using Tavily (AI-optimized, requires API key).

    Tavily is purpose-built for AI applications. Unlike DuckDuckGo which
    returns short snippets, Tavily returns pre-extracted page content
    that's optimized for LLM consumption. This means:
      - Higher quality, more relevant content per result
      - Content is already cleaned and formatted
      - Less need to fetch individual URLs separately

    PRICING (as of 2025):
      - Free tier: 1,000 searches/month (no credit card required)
      - Paid: $30/month for 4,000 searches, scaling up
      - See: https://tavily.com/pricing

    Args:
        query: The search query string.
        api_key: Tavily API key.

    Returns:
        List of SearchResult objects (up to WEB_SEARCH_MAX_RESULTS).
    """
    if not api_key:
        logger.warning("[WebSearch] Tavily API key not provided, falling back to DuckDuckGo")
        return await _search_duckduckgo(query)

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            max_results=WEB_SEARCH_MAX_RESULTS,
            search_depth="basic",
        )

        results: list[SearchResult] = []
        for r in response.get("results", []):
            results.append(
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    content=r.get("content", ""),
                )
            )
        return results

    except ImportError:
        logger.error("[WebSearch] tavily-python not installed. Run: pip install tavily-python")
        return await _search_duckduckgo(query)
    except Exception as e:
        logger.warning(f"[WebSearch] Tavily search failed: {e}. Falling back to DuckDuckGo.")
        return await _search_duckduckgo(query)


# =============================================================================
# SEARCH DISPATCHER
# =============================================================================


async def search_web(
    query: str,
    provider: str = "duckduckgo",
    tavily_api_key: str | None = None,
) -> list[SearchResult]:
    """Search the web using the specified provider.

    This is the main entry point for web search. It dispatches to the
    appropriate provider and handles fallback from Tavily to DuckDuckGo.

    Args:
        query: The search query string.
        provider: "duckduckgo" or "tavily".
        tavily_api_key: API key for Tavily (required if provider is "tavily").

    Returns:
        List of SearchResult objects.
    """
    if not query or not query.strip():
        return []

    logger.info(f"[WebSearch] Searching with {provider}: {query[:80]}...")

    if provider == "tavily" and tavily_api_key:
        return await _search_tavily(query, tavily_api_key)

    return await _search_duckduckgo(query)


# =============================================================================
# RESULT FORMATTING
# =============================================================================


def format_search_context(
    search_results: list[SearchResult],
    fetched_urls: dict[str, str] | None = None,
    query: str = "",
) -> str:
    """Format search results and fetched URL content into a system message.

    This creates the text that gets injected into the conversation as a
    system message. The LLM sees this context and can reference it in
    its response.

    The format is designed to be:
      1. Clearly delimited (so the LLM knows what's search context vs. user input)
      2. Structured (numbered results with title, URL, content)
      3. Instructive (tells the LLM how to use the results)

    Args:
        search_results: List of search results from search_web().
        fetched_urls: Dict mapping URL → extracted content (from fetch_url_content).
        query: The original search query (for the header).

    Returns:
        Formatted system message string.
    """
    parts: list[str] = []

    # Header
    parts.append(f'[Web Search Results for: "{query}"]')
    parts.append("")

    # Fetched URL content (if any URLs were explicitly shared by the user)
    if fetched_urls:
        parts.append("=== Content from URLs shared by the user ===")
        parts.append("")
        for url, content in fetched_urls.items():
            parts.append(f"URL: {url}")
            parts.append(f"Content: {content[:WEB_SEARCH_MAX_CONTENT_LENGTH]}")
            parts.append("")

    # Search results
    if search_results:
        if fetched_urls:
            parts.append("=== Web search results ===")
            parts.append("")
        for i, result in enumerate(search_results, 1):
            parts.append(f"{i}. {result.title}")
            parts.append(f"   URL: {result.url}")
            parts.append(f"   {result.content}")
            parts.append("")

    # Footer with instructions for the LLM
    parts.append("[End of Web Search Results]")
    parts.append("")
    parts.append(
        "Use the above search results to inform your answer. "
        "When citing sources, ALWAYS use markdown link format: [source title](URL). "
        "Never paste raw URLs as plain text — always wrap them in markdown links. "
        "If the search results don't contain relevant information, answer "
        "based on your training knowledge and mention that you couldn't "
        "find relevant web results."
    )

    return "\n".join(parts)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


async def perform_web_search(
    query: str,
    full_message_text: str | None = None,
    provider: str = "duckduckgo",
    tavily_api_key: str | None = None,
) -> tuple[str | None, list[str]]:
    """Perform a complete web search operation: search + URL fetch + format.

    This is the function called by the chat route handler. It orchestrates
    the full web search pipeline:
      1. Detect URLs in the user's message and fetch their content
      2. Search the web for the user's query
      3. Format everything into a system message

    Args:
        query: The user's query text (used as the search query).
        full_message_text: The full text of the user's message (for URL detection).
            If None, uses query.
        provider: Search provider ("duckduckgo" or "tavily").
        tavily_api_key: API key for Tavily (if using Tavily provider).

    Returns:
        Tuple of:
          - Formatted system message string (or None if everything failed)
          - List of source URLs (for frontend display)
    """
    message_text = full_message_text or query
    source_urls: list[str] = []

    # Step 1: Detect and fetch URLs from the user's message
    fetched_urls: dict[str, str] = {}
    detected_urls = extract_urls(message_text)
    if detected_urls:
        logger.info(f"[WebSearch] Found {len(detected_urls)} URL(s) in message")
        for url in detected_urls[:3]:  # Limit to 3 URLs to avoid excessive fetching
            content = await fetch_url_content(url)
            if content:
                fetched_urls[url] = content
                source_urls.append(url)

    # Step 2: Search the web
    search_results = await search_web(query, provider, tavily_api_key)
    for result in search_results:
        if result.url:
            source_urls.append(result.url)

    # Step 3: Format results
    if not search_results and not fetched_urls:
        logger.info("[WebSearch] No results found")
        return None, []

    context = format_search_context(search_results, fetched_urls or None, query)
    return context, source_urls
