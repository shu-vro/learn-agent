from langchain_community.tools import DuckDuckGoSearchRun, YouTubeSearchTool

duckduckgo_search = DuckDuckGoSearchRun()
youtube_search = YouTubeSearchTool()

__all__ = ["duckduckgo_search", "youtube_search"]
