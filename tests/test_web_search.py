"""
Tests for web search functionality.

This module validates the web search pipeline that enables STREAM to
augment LLM responses with current information from the internet.

Tests cover:
  - URL detection and extraction from user messages
  - URL content fetching with error handling
  - DuckDuckGo and Tavily search provider integration
  - Search result formatting for LLM context injection
  - The full perform_web_search orchestration pipeline
  - Provider fallback behavior (Tavily → DuckDuckGo)
  - Edge cases (empty queries, no results, timeouts)
  - Backend request model validation (web search fields)
  - Config constants validation

These tests use mocking to avoid making real HTTP/search requests,
so they run fast and don't require network access or API keys.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from stream.middleware.utils.web_search import (
    SearchResult,
    extract_urls,
    fetch_url_content,
    format_search_context,
    perform_web_search,
    search_web,
)

# =============================================================================
# URL EXTRACTION TESTS
# =============================================================================


class TestExtractUrls:
    """Tests for the extract_urls() function."""

    def test_single_http_url(self):
        text = "Check out http://example.com for more info"
        assert extract_urls(text) == ["http://example.com"]

    def test_single_https_url(self):
        text = "Visit https://example.com/page for details"
        assert extract_urls(text) == ["https://example.com/page"]

    def test_multiple_urls(self):
        text = "See https://one.com and https://two.org for info"
        result = extract_urls(text)
        assert result == ["https://one.com", "https://two.org"]

    def test_no_urls(self):
        assert extract_urls("No URLs here, just text.") == []

    def test_empty_string(self):
        assert extract_urls("") == []

    def test_url_with_path_and_query(self):
        text = "https://example.com/path/to/page?q=search&lang=en"
        result = extract_urls(text)
        assert len(result) == 1
        assert "example.com/path/to/page" in result[0]

    def test_url_with_trailing_punctuation(self):
        """URLs at the end of sentences shouldn't include the period."""
        text = "Visit https://example.com."
        result = extract_urls(text)
        assert result == ["https://example.com"]

    def test_url_with_trailing_comma(self):
        text = "See https://example.com, https://other.org for info"
        result = extract_urls(text)
        assert result == ["https://example.com", "https://other.org"]

    def test_duplicate_urls_deduplicated(self):
        text = "Visit https://example.com and again https://example.com"
        result = extract_urls(text)
        assert result == ["https://example.com"]

    def test_url_with_fragment(self):
        text = "See https://example.com/page#section for details"
        result = extract_urls(text)
        assert len(result) == 1
        assert "#section" in result[0]

    def test_url_in_parentheses(self):
        text = "More info (https://example.com/docs)"
        result = extract_urls(text)
        assert result == ["https://example.com/docs"]


# =============================================================================
# URL CONTENT FETCHING TESTS
# =============================================================================


class TestFetchUrlContent:
    """Tests for the fetch_url_content() function."""

    @pytest.mark.asyncio
    async def test_fetch_html_content(self):
        """Fetching an HTML page should return extracted text."""
        html = "<html><body><p>Hello world</p><script>var x=1;</script></body></html>"
        mock_response = MagicMock()
        mock_response.text = html
        mock_response.headers = {"content-type": "text/html"}
        mock_response.raise_for_status = MagicMock()

        with patch("stream.middleware.utils.web_search.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get.return_value = mock_response
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_instance

            result = await fetch_url_content("https://example.com")

        assert result is not None
        assert "Hello world" in result
        # Script content should be removed by BeautifulSoup
        assert "var x=1" not in result

    @pytest.mark.asyncio
    async def test_fetch_plain_text(self):
        """Fetching a plain text URL should return the raw text."""
        mock_response = MagicMock()
        mock_response.text = "Plain text content here"
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.raise_for_status = MagicMock()

        with patch("stream.middleware.utils.web_search.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get.return_value = mock_response
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_instance

            result = await fetch_url_content("https://example.com/file.txt")

        assert result == "Plain text content here"

    @pytest.mark.asyncio
    async def test_fetch_non_text_content_returns_none(self):
        """Binary content types (images, PDFs) should return None."""
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "application/pdf"}
        mock_response.raise_for_status = MagicMock()

        with patch("stream.middleware.utils.web_search.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get.return_value = mock_response
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_instance

            result = await fetch_url_content("https://example.com/file.pdf")

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_timeout_returns_none(self):
        """Timeout errors should return None gracefully."""
        import httpx

        with patch("stream.middleware.utils.web_search.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get.side_effect = httpx.TimeoutException("timed out")
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_instance

            result = await fetch_url_content("https://slow-site.com")

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_http_error_returns_none(self):
        """HTTP errors (404, 500) should return None gracefully."""
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=mock_response
        )

        with patch("stream.middleware.utils.web_search.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get.return_value = mock_response
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_instance

            result = await fetch_url_content("https://example.com/missing")

        assert result is None


# =============================================================================
# DUCKDUCKGO SEARCH TESTS
# =============================================================================


class TestDuckDuckGoSearch:
    """Tests for the DuckDuckGo search provider."""

    @pytest.mark.asyncio
    async def test_duckduckgo_returns_results(self):
        """DuckDuckGo search should return SearchResult objects."""
        mock_results = [
            {"title": "Python Docs", "href": "https://python.org", "body": "Official Python docs"},
            {"title": "Tutorial", "href": "https://tutorial.com", "body": "Learn Python"},
        ]

        with patch("duckduckgo_search.DDGS") as mock_ddgs_class:
            mock_ddgs = MagicMock()
            mock_ddgs.text.return_value = mock_results
            mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
            mock_ddgs.__exit__ = MagicMock(return_value=False)
            mock_ddgs_class.return_value = mock_ddgs

            results = await search_web("python tutorial", provider="duckduckgo")

        assert len(results) == 2
        assert results[0].title == "Python Docs"
        assert results[0].url == "https://python.org"
        assert results[1].title == "Tutorial"

    @pytest.mark.asyncio
    async def test_duckduckgo_handles_exception(self):
        """DuckDuckGo errors should return an empty list, not raise."""
        with patch("duckduckgo_search.DDGS") as mock_ddgs_class:
            mock_ddgs_class.side_effect = Exception("Rate limited")

            results = await search_web("test query", provider="duckduckgo")

        assert results == []

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self):
        """Empty or whitespace-only queries should short-circuit."""
        results = await search_web("", provider="duckduckgo")
        assert results == []

        results = await search_web("   ", provider="duckduckgo")
        assert results == []


# =============================================================================
# TAVILY SEARCH TESTS
# =============================================================================


class TestTavilySearch:
    """Tests for the Tavily search provider."""

    @pytest.mark.asyncio
    async def test_tavily_returns_results(self):
        """Tavily search should return SearchResult objects."""
        mock_response = {
            "results": [
                {"title": "AI News", "url": "https://ai.com", "content": "Latest AI developments"},
            ]
        }

        with patch("tavily.TavilyClient") as mock_tavily_class:
            mock_client = MagicMock()
            mock_client.search.return_value = mock_response
            mock_tavily_class.return_value = mock_client

            results = await search_web(
                "AI news",
                provider="tavily",
                tavily_api_key="test-key-123",
            )

        assert len(results) == 1
        assert results[0].title == "AI News"
        assert results[0].url == "https://ai.com"

    @pytest.mark.asyncio
    async def test_tavily_without_key_falls_back_to_duckduckgo(self):
        """Tavily without an API key should fall back to DuckDuckGo."""
        mock_results = [
            {"title": "Fallback Result", "href": "https://fallback.com", "body": "DDG result"},
        ]

        with patch("duckduckgo_search.DDGS") as mock_ddgs_class:
            mock_ddgs = MagicMock()
            mock_ddgs.text.return_value = mock_results
            mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
            mock_ddgs.__exit__ = MagicMock(return_value=False)
            mock_ddgs_class.return_value = mock_ddgs

            results = await search_web(
                "test",
                provider="tavily",
                tavily_api_key=None,
            )

        assert len(results) == 1
        assert results[0].title == "Fallback Result"

    @pytest.mark.asyncio
    async def test_tavily_error_falls_back_to_duckduckgo(self):
        """Tavily API errors should fall back to DuckDuckGo."""
        mock_results = [
            {"title": "DDG Fallback", "href": "https://ddg.com", "body": "From DuckDuckGo"},
        ]

        with patch("tavily.TavilyClient") as mock_tavily_class:
            mock_tavily_class.side_effect = Exception("API error")

            with patch("duckduckgo_search.DDGS") as mock_ddgs_class:
                mock_ddgs = MagicMock()
                mock_ddgs.text.return_value = mock_results
                mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
                mock_ddgs.__exit__ = MagicMock(return_value=False)
                mock_ddgs_class.return_value = mock_ddgs

                results = await search_web(
                    "test",
                    provider="tavily",
                    tavily_api_key="bad-key",
                )

        assert len(results) == 1
        assert results[0].title == "DDG Fallback"


# =============================================================================
# GOOGLE SEARCH (SERPER.DEV) TESTS
# =============================================================================


class TestGoogleSearch:
    """Tests for the Google Search provider (via Serper.dev)."""

    @pytest.mark.asyncio
    async def test_google_returns_results(self):
        """Serper.dev search should return SearchResult objects from Google."""
        mock_json = {
            "organic": [
                {
                    "title": "Google Result",
                    "link": "https://example.com/result",
                    "snippet": "A Google result",
                },
            ]
        }
        mock_response = MagicMock()
        mock_response.json.return_value = mock_json
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_class.return_value = mock_client

            results = await search_web(
                "test query",
                provider="google",
                serper_api_key="test-serper-key",
            )

        assert len(results) == 1
        assert results[0].title == "Google Result"
        assert results[0].url == "https://example.com/result"
        assert results[0].content == "A Google result"

    @pytest.mark.asyncio
    async def test_google_without_key_falls_back_to_duckduckgo(self):
        """Google without Serper API key should fall back to DuckDuckGo."""
        mock_results = [
            {"title": "DDG Fallback", "href": "https://ddg.com", "body": "From DuckDuckGo"},
        ]

        with patch("duckduckgo_search.DDGS") as mock_ddgs_class:
            mock_ddgs = MagicMock()
            mock_ddgs.text.return_value = mock_results
            mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
            mock_ddgs.__exit__ = MagicMock(return_value=False)
            mock_ddgs_class.return_value = mock_ddgs

            results = await search_web(
                "test",
                provider="google",
                serper_api_key=None,
            )

        assert len(results) == 1
        assert results[0].title == "DDG Fallback"

    @pytest.mark.asyncio
    async def test_google_error_falls_back_to_duckduckgo(self):
        """Serper API errors should fall back to DuckDuckGo."""
        mock_results = [
            {"title": "DDG Fallback", "href": "https://ddg.com", "body": "From DuckDuckGo"},
        ]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.side_effect = Exception("API quota exceeded")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_class.return_value = mock_client

            with patch("duckduckgo_search.DDGS") as mock_ddgs_class:
                mock_ddgs = MagicMock()
                mock_ddgs.text.return_value = mock_results
                mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
                mock_ddgs.__exit__ = MagicMock(return_value=False)
                mock_ddgs_class.return_value = mock_ddgs

                results = await search_web(
                    "test",
                    provider="google",
                    serper_api_key="bad-key",
                )

        assert len(results) == 1
        assert results[0].title == "DDG Fallback"


# =============================================================================
# RESULT FORMATTING TESTS
# =============================================================================


class TestFormatSearchContext:
    """Tests for the format_search_context() function."""

    def test_formats_search_results(self):
        """Search results should be formatted as numbered list."""
        results = [
            SearchResult("Title One", "https://one.com", "Content one"),
            SearchResult("Title Two", "https://two.com", "Content two"),
        ]

        context = format_search_context(results, query="test query")

        assert '[Web Search Results for: "test query"]' in context
        assert "1. Title One" in context
        assert "URL: https://one.com" in context
        assert "Content one" in context
        assert "2. Title Two" in context
        assert "[End of Web Search Results]" in context
        assert "citing sources" in context

    def test_formats_fetched_urls(self):
        """Fetched URL content should appear before search results."""
        results = [SearchResult("MySearchResult", "https://search.com", "Searched content")]
        fetched = {"https://user-shared.com": "Fetched page content here"}

        context = format_search_context(results, fetched_urls=fetched, query="test")

        assert "Content from URLs shared by the user" in context
        assert "https://user-shared.com" in context
        assert "Fetched page content here" in context
        # Search results should come after fetched URLs
        url_pos = context.find("user-shared.com")
        search_pos = context.find("MySearchResult")
        assert url_pos < search_pos

    def test_empty_results_still_has_structure(self):
        """Even with no results, the format should have delimiters."""
        context = format_search_context([], query="no results query")

        assert '[Web Search Results for: "no results query"]' in context
        assert "[End of Web Search Results]" in context

    def test_no_fetched_urls_omits_section(self):
        """When no URLs were fetched, the URL section should not appear."""
        results = [SearchResult("Only Search", "https://s.com", "Content")]
        context = format_search_context(results, query="q")

        assert "Content from URLs shared by the user" not in context


# =============================================================================
# PERFORM_WEB_SEARCH ORCHESTRATION TESTS
# =============================================================================


class TestPerformWebSearch:
    """Tests for the perform_web_search() orchestration function."""

    @pytest.mark.asyncio
    async def test_returns_formatted_context_and_sources(self):
        """Full pipeline should return context string and source URLs."""
        mock_results = [
            {"title": "Result", "href": "https://result.com", "body": "Found it"},
        ]

        with patch("duckduckgo_search.DDGS") as mock_ddgs_class:
            mock_ddgs = MagicMock()
            mock_ddgs.text.return_value = mock_results
            mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
            mock_ddgs.__exit__ = MagicMock(return_value=False)
            mock_ddgs_class.return_value = mock_ddgs

            context, sources = await perform_web_search("test query")

        assert context is not None
        assert "Result" in context
        assert "https://result.com" in sources

    @pytest.mark.asyncio
    async def test_detects_and_fetches_urls_in_message(self):
        """URLs in the user's message should be detected and fetched."""
        html = "<html><body><p>Page content</p></body></html>"
        mock_response = MagicMock()
        mock_response.text = html
        mock_response.headers = {"content-type": "text/html"}
        mock_response.raise_for_status = MagicMock()

        mock_search_results = [
            {"title": "Search", "href": "https://search.com", "body": "Searched"},
        ]

        with (
            patch("stream.middleware.utils.web_search.httpx.AsyncClient") as mock_client,
            patch("duckduckgo_search.DDGS") as mock_ddgs_class,
        ):
            mock_instance = AsyncMock()
            mock_instance.get.return_value = mock_response
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_instance

            mock_ddgs = MagicMock()
            mock_ddgs.text.return_value = mock_search_results
            mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
            mock_ddgs.__exit__ = MagicMock(return_value=False)
            mock_ddgs_class.return_value = mock_ddgs

            context, sources = await perform_web_search(
                query="Summarize this",
                full_message_text="Summarize this: https://article.com/post",
            )

        assert context is not None
        assert "Page content" in context
        assert "https://article.com/post" in sources

    @pytest.mark.asyncio
    async def test_no_results_returns_none(self):
        """When search returns nothing and no URLs found, return None."""
        with patch("duckduckgo_search.DDGS") as mock_ddgs_class:
            mock_ddgs = MagicMock()
            mock_ddgs.text.return_value = []
            mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
            mock_ddgs.__exit__ = MagicMock(return_value=False)
            mock_ddgs_class.return_value = mock_ddgs

            context, sources = await perform_web_search("obscure query xyz")

        assert context is None
        assert sources == []


# =============================================================================
# BACKEND REQUEST MODEL TESTS
# =============================================================================


class TestChatCompletionRequestWebSearch:
    """Verify that the ChatCompletionRequest model accepts web search fields."""

    def test_web_search_fields_have_defaults(self):
        """web_search, web_search_provider, and API keys should have safe defaults."""
        from stream.middleware.routes.chat import ChatCompletionRequest

        req = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert req.web_search is False
        assert req.web_search_provider == "duckduckgo"
        assert req.tavily_api_key is None
        assert req.serper_api_key is None

    def test_web_search_tavily_enabled(self):
        """Tavily web search fields should be settable."""
        from stream.middleware.routes.chat import ChatCompletionRequest

        req = ChatCompletionRequest(
            messages=[{"role": "user", "content": "What's new in Python?"}],
            web_search=True,
            web_search_provider="tavily",
            tavily_api_key="tvly-test-key",
        )
        assert req.web_search is True
        assert req.web_search_provider == "tavily"
        assert req.tavily_api_key == "tvly-test-key"

    def test_web_search_google_enabled(self):
        """Google (Serper) web search fields should be settable."""
        from stream.middleware.routes.chat import ChatCompletionRequest

        req = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Latest AI trends"}],
            web_search=True,
            web_search_provider="google",
            serper_api_key="test-serper-key",
        )
        assert req.web_search is True
        assert req.web_search_provider == "google"
        assert req.serper_api_key == "test-serper-key"


# =============================================================================
# CONFIG CONSTANTS TESTS
# =============================================================================


class TestWebSearchConfig:
    """Validate web search configuration constants."""

    def test_max_results_is_reasonable(self):
        from stream.middleware.config import WEB_SEARCH_MAX_RESULTS

        assert 1 <= WEB_SEARCH_MAX_RESULTS <= 20

    def test_max_content_length_is_reasonable(self):
        from stream.middleware.config import WEB_SEARCH_MAX_CONTENT_LENGTH

        assert 500 <= WEB_SEARCH_MAX_CONTENT_LENGTH <= 10000

    def test_timeout_is_reasonable(self):
        from stream.middleware.config import WEB_SEARCH_TIMEOUT

        assert 1 <= WEB_SEARCH_TIMEOUT <= 30


# =============================================================================
# SEARCH RESULT DATACLASS TESTS
# =============================================================================


class TestSearchResult:
    """Tests for the SearchResult data class."""

    def test_creation(self):
        result = SearchResult("Title", "https://url.com", "Content")
        assert result.title == "Title"
        assert result.url == "https://url.com"
        assert result.content == "Content"

    def test_repr(self):
        result = SearchResult("Title", "https://url.com", "Content")
        r = repr(result)
        assert "Title" in r
        assert "url.com" in r


# =============================================================================
# DESKTOP BUILD SPEC TESTS
# =============================================================================


class TestDesktopBuildSpec:
    """Verify that stream.spec includes web search dependencies."""

    def test_spec_includes_web_search_hiddenimports(self):
        """The PyInstaller spec file should list web search packages."""
        import pathlib

        spec_path = pathlib.Path(__file__).parent.parent / "stream.spec"
        spec_content = spec_path.read_text()

        assert "duckduckgo_search" in spec_content
        assert "bs4" in spec_content
        assert "lxml" in spec_content
        assert "tavily" in spec_content


# =============================================================================
# PYPROJECT.TOML DEPENDENCY TESTS
# =============================================================================


class TestProjectDependencies:
    """Verify that web search dependencies are declared in pyproject.toml."""

    def test_pyproject_includes_web_search_deps(self):
        import pathlib

        pyproject_path = pathlib.Path(__file__).parent.parent / "pyproject.toml"
        content = pyproject_path.read_text()

        assert "duckduckgo-search" in content
        assert "beautifulsoup4" in content
        assert "lxml" in content
        assert "tavily-python" in content
