import asyncio
from mem0 import AsyncMemory, Memory
from mem0.configs.base import MemoryConfig
from src.config.mem0_config import config


custom_config = MemoryConfig(**config)
memory = Memory(config=custom_config)


async def _with_timeout_and_retry(operation, max_retries=3, timeout=10.0):
    for attempt in range(max_retries):
        try:
            return await asyncio.wait_for(operation(), timeout=timeout)
        except asyncio.TimeoutError:
            print(f"Timeout on attempt {attempt + 1}")
        except Exception as exc:
            print(f"Error on attempt {attempt + 1}: {exc}")

        if attempt < max_retries - 1:
            await asyncio.sleep(2**attempt)

    raise Exception(f"Operation failed after {max_retries} attempts")


async def memory_search(
    query: str,
    user_id: str = "default_user",
    filters: dict = None,
    limit: int = 5,
):
    async def search_operation():
        return await memory.search(query, user_id=user_id, filters=filters, limit=limit)

    return await _with_timeout_and_retry(search_operation)


async def memory_add(
    content: str,
    metadata: dict,
    user_id: str = "default_user",
):
    async def add_operation():
        await memory.add(content, user_id=user_id, metadata=metadata)

    await _with_timeout_and_retry(add_operation)
