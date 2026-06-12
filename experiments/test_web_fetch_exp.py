from pathlib import Path
import sys

from dotenv import load_dotenv
from langchain.agents import create_agent

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from src.config.model_config import model_selector
from src.config.constants import DEFAULT_LLM_MODEL
import src.config.bootstrap  # noqa: F401

from src.agent.tools.web_fetch import fetch_url

url = "https://shirshen.is-a.dev"
# content = fetch_url(url)
# print(content)

llm = model_selector(DEFAULT_LLM_MODEL)
agent = create_agent(
    model=llm,
    tools=[fetch_url],
    system_prompt="You are a helpful assistant that can fetch web content.",
)

result = agent.invoke({"messages": (f"search this website: {url} what is this about?")})
print(result)
