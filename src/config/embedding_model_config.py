from types import MappingProxyType

EMBEDDING_MODELS = MappingProxyType(
    {
        "hf:Octen/Octen-Embedding-0.6B": {
            "provider": "hf",
            "model": "Octen/Octen-Embedding-0.6B",
            "dimension": 1024,
            "input": 0,
            "output": 0,
            "max_tokens": 32768,
        },
        "ollama:nomic-embed-text": {
            "provider": "ollama",
            "model": "nomic-embed-text",
            "dimension": 1024,
            "input": 0,
            "output": 0,
            "max_tokens": 2048,
        },
        "openai:text-embedding-3-small": {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "dimension": 1536,
            "input": 0.02,
            "output": 0,
            "max_tokens": 8192,
        },
    }
)

__all__ = ["EMBEDDING_MODELS"]
