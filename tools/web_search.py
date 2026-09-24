"""
Web search tool — lets the agent search the internet.
Uses ddgs (the new name for duckduckgo_search).
Free, no API key needed.
"""
from ddgs import DDGS
import time


def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web and return formatted top results.
    Retries once if the first attempt fails (DuckDuckGo is flaky sometimes).
    """
    for attempt in range(2):
        try:
            results = list(DDGS().text(query, max_results=max_results))

            if not results:
                if attempt == 0:
                    time.sleep(2)
                    continue
                return f"No results found for: {query}"

            output = []
            for i, r in enumerate(results, 1):
                output.append(
                    f"{i}. {r.get('title', 'No title')}\n"
                    f"   {r.get('body', 'No description')}\n"
                    f"   Source: {r.get('href', 'No URL')}\n"
                )
            return "\n".join(output)

        except Exception as e:
            if attempt == 0:
                time.sleep(2)
                continue
            return f"Search failed: {str(e)}"

    return f"No results found for: {query}"