# Web Search — Internet Connectivity for STREAM

## Overview

STREAM can augment LLM responses with current information from the internet. When web search is enabled, the backend searches the web for the user's query **before** sending it to the LLM. Search results are injected as a system message, giving the LLM access to up-to-date information it wouldn't otherwise have.

This feature works across **all three tiers** (Local, Lakeshore, Cloud) because it uses plain text context injection — no tool calling or function calling support required from the model.

### Key Properties

| Property | Value |
|---|---|
| **Default state** | Off (user must enable via globe toggle) |
| **Default provider** | DuckDuckGo (free, no API key) |
| **Premium providers** | Tavily (AI-optimized) · Google Custom Search (highest quality) |
| **Toggle location** | Globe icon in the chat input area |
| **Provider config** | Advanced Settings in the sidebar |
| **Works with** | All models across all tiers |

---

## How It Works

### Architecture

```
User sends message
    │
    ▼
┌──────────────────┐
│ Web search        │  ← Only if globe toggle is ON
│ enabled?          │
└────────┬─────────┘
         │ Yes
         ▼
┌──────────────────┐     ┌──────────────────┐
│ Detect URLs in   │────▶│ Fetch URL content │
│ user's message   │     │ (with BeautifulSoup)
└────────┬─────────┘     └────────┬─────────┘
         │                        │
         ▼                        │
┌──────────────────┐              │
│ Search the web   │              │
│ (DDG/Tavily/Google)             │
└────────┬─────────┘              │
         │                        │
         ▼                        │
┌──────────────────┐◀─────────────┘
│ Format results   │
│ as system message│
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Prepend to       │
│ conversation     │
│ messages         │
└────────┬─────────┘
         │
         ▼
   Normal LLM routing
   (Local / Lakeshore / Cloud)
```

### Step-by-Step Flow

1. **User enables web search** by clicking the globe icon in the chat input area. The globe turns blue and shows a "Web" label.

2. **User sends a message** (e.g., "What are the latest features in Python 3.13?").

3. **Backend receives the request** with `web_search: true` in the payload.

4. **URL detection**: If the user's message contains URLs (e.g., "summarize this: https://article.com"), those URLs are fetched and their content is extracted using BeautifulSoup.

5. **Web search**: The user's query is sent to the selected search provider (DuckDuckGo, Tavily, or Google). The provider returns up to 5 results with titles, URLs, and content snippets.

6. **Result formatting**: All search results and fetched URL content are formatted into a structured system message with clear delimiters.

7. **Context injection**: The formatted system message is **prepended** to the conversation messages. The LLM sees it as context and can reference it in its response.

8. **Normal routing**: The augmented conversation goes through the normal tier routing (complexity analysis, model selection, streaming).

9. **Source URLs**: The metadata SSE event includes `web_search_sources` — a list of URLs that the frontend can optionally display.

### Why Pre-Query Injection (Not Tool Calling)

Tool calling requires the model to:
1. Recognize it needs to search
2. Emit a `tool_call` response
3. Wait for the tool result
4. Generate a final response

This multi-turn dance requires specific model support. Many models — especially small ones like Llama 3.2 3B running locally on Ollama — don't support tool calling. By injecting search results **before** the LLM call, we make web search work with every model across every tier.

---

## Providers

### DuckDuckGo (Default)

| Property | Details |
|---|---|
| **Cost** | Free |
| **API key** | Not required |
| **Library** | `duckduckgo-search` (Python) |
| **How it works** | Scrapes DuckDuckGo search results |
| **Result quality** | Good for general queries |
| **Rate limiting** | May throttle heavy usage from a single IP |
| **Best for** | Desktop mode, campus use, zero-config setups |
| **Privacy** | DuckDuckGo doesn't track searches |

DuckDuckGo is the default provider because:
- It requires no API key (works immediately after installation)
- It's free with no usage limits
- It respects user privacy
- It works in desktop mode where users don't want to manage API keys

**Limitation**: DuckDuckGo returns short content snippets (~200 characters per result). For richer content, STREAM would need to fetch each result's URL separately, which adds latency. For research-heavy queries, Tavily provides better content extraction.

### Tavily (Optional Premium)

| Property | Details |
|---|---|
| **Cost** | Free tier: 1,000 searches/month; Paid: from $30/month |
| **API key** | Required (get one at [tavily.com](https://tavily.com)) |
| **Library** | `tavily-python` |
| **How it works** | Dedicated AI-optimized search API |
| **Result quality** | Higher — pre-extracted content designed for LLMs |
| **Rate limiting** | Based on plan (1K/month free) |
| **Best for** | Server deployments, research queries, higher quality needs |

Tavily is the premium alternative because:
- It returns pre-extracted page content (not just snippets)
- Content is cleaned and formatted for LLM consumption
- Results are more relevant for complex research queries
- Built-in URL content extraction (less need for separate fetching)

**Fallback behavior**: If Tavily fails (API error, invalid key, service down), STREAM automatically falls back to DuckDuckGo. This ensures web search never completely fails.

### Google Search via Serper.dev (Optional Premium)

| Property | Details |
|---|---|
| **Cost** | Free: 2,500 queries (one-time); Paid: $50 for 50,000 ($1/1K) |
| **API key** | Required (sign up at [serper.dev](https://serper.dev)) |
| **How it works** | Serper.dev API returns real-time Google search results |
| **Result quality** | Highest — actual Google search results from google.com |
| **Rate limiting** | Based on purchased credits |
| **Best for** | Users who want Google-quality search results |

Serper.dev provides actual Google search results through a simple REST API. Unlike Google's deprecated Programmable Search Engine (which only searches specific sites you configure), Serper searches the entire web using Google's full index.

**Why Serper instead of Google Custom Search?** Google deprecated the "Search the entire web" feature for new Programmable Search Engines in 2025–2026. New engines can only search specific sites you add, making it useless for general web search. Serper.dev solves this by providing real Google results through a clean API with just one API key.

**Setup**: Sign up at [serper.dev](https://serper.dev), copy your API key — that's it. No Search Engine ID or Google Cloud Console needed.

**Pricing reference**: https://serper.dev

**Fallback behavior**: If Serper fails (API error, invalid key, credits exhausted), STREAM automatically falls back to DuckDuckGo.

---

## Configuration

### User-Facing Controls

#### Globe Toggle (Chat Input)

The globe icon sits between the camera button and the text input area:

```
┌──────────────────────────────────────────┐
│ [Upload] [Camera] [Globe]  │ Type...  │ [Send] │
└──────────────────────────────────────────┘
```

- **Click** to toggle web search on/off for the current message
- **Blue with "Web" label** = web search is active
- **Gray** = web search is off (default)
- The toggle state resets each session (doesn't persist)

#### Provider Selection (Advanced Settings)

In the sidebar under **Advanced Settings > Web Search Provider**:

- **DuckDuckGo**: Selected by default. No configuration needed. Free with no limits.
- **Tavily**: When selected, an API key input field appears. 1,000 free searches/month. Paid plans from $30/month.
- **Google Search**: When selected, an API key input field appears. Powered by Serper.dev. 2,500 free queries. Paid: $50 for 50K.

All API keys are stored in localStorage (persists across sessions) and only sent to the backend when the corresponding provider is selected.

### Backend Configuration

Constants in `stream/middleware/config.py`:

| Constant | Default | Description |
|---|---|---|
| `WEB_SEARCH_MAX_RESULTS` | 5 | Number of search results to include |
| `WEB_SEARCH_MAX_CONTENT_LENGTH` | 4000 | Max characters per fetched URL |
| `WEB_SEARCH_TIMEOUT` | 10 | Timeout for search/fetch operations (seconds) |

### Request Payload

The frontend sends these additional fields in the chat request:

```json
{
  "web_search": true,
  "web_search_provider": "duckduckgo",
  "tavily_api_key": "tvly-...",
  "serper_api_key": "your-serper-key"
}
```

All fields are optional and have safe defaults:
- `web_search`: `false`
- `web_search_provider`: `"duckduckgo"`
- `tavily_api_key`: `null`
- `serper_api_key`: `null`

---

## Context Window Impact

Web search results consume context window tokens:

| Scenario | Approximate Token Cost |
|---|---|
| 5 search results (snippets only) | 500–1,000 tokens |
| 5 search results + 1 fetched URL | 1,000–2,000 tokens |
| 5 search results + 3 fetched URLs | 2,000–5,000 tokens |

STREAM's existing context window validation (in `chat.py`) catches cases where the conversation + search results exceed the model's context limit. The user will see a "context too long" error with a suggestion to start a new conversation.

### Lakeshore Payload Consideration

Web search results add ~2–4 KB of text to the message payload. This is negligible compared to the 8 MB Globus Compute payload limit (images are typically 100 KB–6 MB). No special handling is needed for Lakeshore.

---

## URL Auto-Detection

When the user's message contains URLs, STREAM automatically:

1. **Detects** URLs using a regex pattern matching `http://` and `https://` schemes
2. **Fetches** up to 3 URLs from the message (to avoid excessive fetching)
3. **Extracts** readable text using BeautifulSoup (strips scripts, styles, nav elements)
4. **Truncates** content to `WEB_SEARCH_MAX_CONTENT_LENGTH` characters
5. **Includes** the extracted content in the system message, before search results

This means users can paste a URL and ask "summarize this" even without knowing about the web search feature — though web search must be enabled for URL fetching to work.

---

## Injected System Message Format

```
[Web Search Results for: "user's query here"]

=== Content from URLs shared by the user ===

URL: https://user-pasted-url.com
Content: Extracted page content...

=== Web search results ===

1. Title of Result 1
   URL: https://example.com/page1
   Content: Snippet from the search result...

2. Title of Result 2
   URL: https://example.com/page2
   Content: Snippet from the search result...

[End of Web Search Results]

Use the above search results to inform your answer.
Cite sources with their URLs when referencing specific information.
If the search results don't contain relevant information, answer
based on your training knowledge and mention that you couldn't
find relevant web results.
```

---

## Error Handling

Web search is designed to be **non-blocking** — if it fails, the chat request continues normally without search results. The LLM answers from its training knowledge.

| Error Scenario | Behavior |
|---|---|
| DuckDuckGo rate limited | Returns empty results; LLM answers without web context |
| Tavily API error | Falls back to DuckDuckGo automatically |
| Tavily key missing | Falls back to DuckDuckGo automatically |
| Serper API error / credits exhausted | Falls back to DuckDuckGo automatically |
| Serper key missing | Falls back to DuckDuckGo automatically |
| URL fetch timeout | Skips that URL; other URLs and search results still work |
| URL returns non-text (PDF, image) | Skips that URL |
| All search + fetch fails | `perform_web_search` returns `None`; chat proceeds normally |
| Search results exceed context | Existing context window validation catches it |

---

## Code File Reference

| File | Purpose |
|---|---|
| `stream/middleware/utils/web_search.py` | Core web search logic: providers, URL fetching, formatting |
| `stream/middleware/config.py` | Configuration constants (`WEB_SEARCH_*`) |
| `stream/middleware/routes/chat.py` | Request model (`web_search` fields) and injection point |
| `stream/middleware/core/streaming.py` | Passes `web_search_sources` in SSE metadata |
| `frontends/react/src/types/settings.ts` | `WebSearchProvider` type definition |
| `frontends/react/src/stores/settingsStore.ts` | Web search state, actions, migrations v7–v8 |
| `frontends/react/src/components/input/ChatInput.tsx` | Globe toggle icon |
| `frontends/react/src/api/stream.ts` | Sends web search fields in API request |
| `frontends/react/src/components/sidebar/SettingsPanel.tsx` | Provider selection, API key inputs (Tavily, Google) |
| `stream.spec` | Desktop build hidden imports for web search packages |
| `pyproject.toml` | Python dependencies for web search |
| `tests/test_web_search.py` | Comprehensive test suite |

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `duckduckgo-search` | >=8.0.0 | DuckDuckGo search provider |
| `tavily-python` | >=0.5.0 | Tavily AI search provider |
| `beautifulsoup4` | >=4.12.0 | HTML content extraction |
| `lxml` | >=5.0.0 | Fast HTML parser for BeautifulSoup |

All are listed in `pyproject.toml` under `[project.dependencies]`.
