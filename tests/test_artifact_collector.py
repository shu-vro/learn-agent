from src.agent.artifact_collector import collect_artifacts


def test_document_chunks_become_artifacts_ignoring_previous_reference_id():
    tools = [
        {
            "name": "retrieve_context",
            "args": {"query": "scaled dot product"},
            "result": (
                "[Source 1] reference_id=abc123:8, type=text, "
                "similarity_score=0.35, page=4\nbody text\n\n"
                "[Source 2] reference_id=abc123:9, type=text, "
                "similarity_score=0.31, page=4\nmore text"
            ),
        },
        {
            "name": "next_chunk",
            "args": {"doc_id": "abc123", "chunk_id": 9},
            "result": (
                "[Next chunk] reference_id=abc123:10, type=text, "
                "previous_reference_id=abc123:9\ntail"
            ),
        },
    ]
    artifacts = collect_artifacts(
        tools, answer="cited [Source 1](reference_id=abc123:8)"
    )
    urls = [a["artifact_url"] for a in artifacts]
    assert urls == [
        "reference_id=abc123:8",
        "reference_id=abc123:9",
        "reference_id=abc123:10",
    ]
    assert all(a["artifact_type"] == "document" for a in artifacts)
    assert artifacts[0]["artifact_metadata"] == {
        "doc_id": "abc123",
        "chunk_id": 8,
        "chunk_type": "text",
        "score": "0.35",
        "page": "4",
        "tool": "retrieve_context",
        "cited": True,
    }
    assert artifacts[1]["artifact_metadata"]["cited"] is False


def test_web_video_and_image_tools():
    tools = [
        {
            "name": "duckduckgo_search",
            "args": {"query": "transformers"},
            "result": (
                "Search hits:\n"
                "- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)\n"
                "  snippet: ...\n\n"
                "[Web 1] [Attention](https://arxiv.org/abs/1706.03762), bm25_score=1\n"
                "excerpt body with an unrelated [ad](https://spam.example.com) link"
            ),
        },
        {
            "name": "fetch_url",
            "args": {"url": "https://example.com/docs"},
            "result": "# page",
        },
        {
            "name": "youtube_search",
            "args": {"query": "transformers, 2"},
            "result": "['https://www.youtube.com/watch?v=aaa', 'https://www.youtube.com/watch?v=bbb']",
        },
        {
            "name": "duckduckgo_image_search",
            "args": {"query": "transformer architecture"},
            "result": "1. Arch\n   url: https://img.example.com/a.png\n   markdown: ![Arch](https://img.example.com/a.png)",
        },
        {"name": "unknown_tool", "args": {}, "result": "https://ignored.example.com"},
    ]
    by_url = {a["artifact_url"]: a for a in collect_artifacts(tools)}
    assert by_url["https://arxiv.org/abs/1706.03762"]["artifact_type"] == "website"
    assert by_url["https://arxiv.org/abs/1706.03762"]["artifact_metadata"]["title"] == (
        "Attention Is All You Need"
    )
    assert "https://spam.example.com" not in by_url  # excerpt links are not artifacts
    assert by_url["https://example.com/docs"]["artifact_type"] == "website"
    assert by_url["https://www.youtube.com/watch?v=bbb"]["artifact_type"] == "video"
    assert by_url["https://img.example.com/a.png"]["artifact_type"] == "image"
    assert "https://ignored.example.com" not in by_url
