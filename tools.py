"""
CONCEPT: Tools
--------------
Three tools bound to the LLM:

1. search_web         - scrapes DuckDuckGo's HTML results page. No API key,
                         no rate-limit headaches — good for a learning project.
2. scrape_page         - fetches a real URL (e.g. one returned by search_web)
                         and pulls out readable text. This is the literal
                         "web scraping" piece.
3. propose_preference  - a *deliberately inert* tool. The LLM calls it when
                         it detects the user stating a food like/dislike, but
                         its Python body never actually runs — the graph
                         intercepts this specific tool call and routes to a
                         human-in-the-loop review node instead (see graph.py).
                         This is a common real pattern: give the model a
                         tool whose *call* is the signal, and let a graph
                         node decide what to do with it.
"""
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool

HEADERS = {"User-Agent": "Mozilla/5.0 (LangGraph learning project)"}


@tool
def search_web(search_query: str) -> str:
    """Run a DuckDuckGo web search for healthy recipes, meal ideas, ingredients
    or nutrition facts. Returns up to 5 results, each with a title, a short
    snippet and its URL. Use this first when you need information but don't
    yet have a specific page to read; then pass the best URL to scrape_page.
    Keep search_query short and specific (e.g. 'high protein vegetarian lunch')."""
    try:
        resp = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": search_query},
            headers=HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        return f"Search failed: {e}"

    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    for block in soup.select(".result__body")[:5]:
        title_tag = block.select_one(".result__a")
        snippet_tag = block.select_one(".result__snippet")
        if not title_tag:
            continue
        title = title_tag.get_text(strip=True)
        url = title_tag.get("href", "")
        snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
        results.append(f"- {title}\n  {snippet}\n  {url}")

    if not results:
        return f"No results found for '{search_query}'."
    return "\n\n".join(results)


@tool
def scrape_page(page_url: str) -> str:
    """Download a web page and return its main readable text (headings, list
    items and paragraphs, truncated to about 3000 characters). Use this on a
    URL returned by search_web to read the actual recipe or article, e.g. to
    extract ingredients, steps or nutrition values before answering.
    page_url must be a full http(s) URL."""
    try:
        resp = requests.get(page_url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        return f"Could not fetch {page_url}: {e}"

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    chunks = [
        el.get_text(" ", strip=True)
        for el in soup.find_all(["h1", "h2", "h3", "li", "p"])
    ]
    text = "\n".join(c for c in chunks if len(c) > 20)
    return text[:3000] if text else "No readable content found on that page."


@tool
def propose_preference(food_item: str, preference_type: str) -> str:
    """Suggest saving a food preference to the user's long-term memory.
    Call this only when the user explicitly says they like or dislike a
    specific food or ingredient (e.g. 'I love avocado', 'I can't stand tofu').
    food_item: the food or ingredient name, e.g. 'avocado'.
    preference_type: exactly 'likes' or 'dislikes'.
    A human must approve before anything is saved, so call this tool on its
    own — never together with other tools in the same turn."""
    # Never actually executed: graph.py routes this call to a
    # human-in-the-loop node instead of the normal tool executor.
    return "pending human review"


SEARCH_TOOLS = [search_web, scrape_page]
ALL_TOOLS = [search_web, scrape_page, propose_preference]
