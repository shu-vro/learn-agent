from langchain_community.tools import DuckDuckGoSearchResults, YouTubeSearchTool

duckduckgo_search = DuckDuckGoSearchResults(
    num_results=5,
    output_format="string",
    name="duckduckgo_search",
    description=(
        "Search the public web via DuckDuckGo. "
        "Returns title, snippet, and link for each result. "
        "Input should be a search query."
    ),
)
youtube_search = YouTubeSearchTool()

__all__ = ["duckduckgo_search", "youtube_search"]
