from types import MappingProxyType

EMBEDDING_MODELS = MappingProxyType(
    {
        "hf:Octen/Octen-Embedding-0.6B": MappingProxyType(
            {
                "provider": "hf",
                "model": "Octen/Octen-Embedding-0.6B",
                "dimension": 1024,
                "input": 0,
                "output": 0,
                "max_tokens": 32768,
            }
        ),
        "ollama:nomic-embed-text": MappingProxyType(
            {
                "provider": "ollama",
                "model": "nomic-embed-text",
                "dimension": 1024,
                "input": 0,
                "output": 0,
                "max_tokens": 2048,
            }
        ),
        "openai:text-embedding-3-small": MappingProxyType(
            {
                "provider": "openai",
                "model": "text-embedding-3-small",
                "dimension": 1536,
                "input": 0.02,
                "output": 0,
                "max_tokens": 8192,
            }
        ),
        "omlx:bge-m3-mlx-fp16": MappingProxyType(
            {
                "provider": "omlx",
                "model": "bge-m3-mlx-fp16",
                "dimension": 1024,
                "input": 0,
                "output": 0,
                "max_tokens": 8192,
            }
        ),
    }
)

__all__ = ["EMBEDDING_MODELS"]
