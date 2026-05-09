from langchain_community.tools import YouTubeSearchTool, DuckDuckGoSearchRun, ShellTool
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.utils.usage_aggregator_callback

global_usage_aggregator = src.utils.usage_aggregator_callback.UsageAggregatorCallback(
    "langchain_community_explore"
)

yt_search_tool = YouTubeSearchTool()
ddg_search_tool = DuckDuckGoSearchRun()
shell_tool = ShellTool()
model = init_chat_model(model="ollama:gemma4:e2b", callbacks=[global_usage_aggregator])

agent = create_agent(
    model=model,
    tools=[yt_search_tool, ddg_search_tool, shell_tool],
)

messages = []


def answer_question():
    last_message = None
    for chunk in agent.stream(
        {"messages": messages},
        stream_mode="values",
    ):
        if "messages" in chunk:
            last_message = chunk["messages"][-1]
            last_message.pretty_print()

    messages.append(last_message)


while 1:
    question = input("> ")
    messages.append(HumanMessage(content=question))

    if question.lower() in ["exit", "quit"]:
        print(
            "\n\n\nExiting conversation. Here are all the messages in the conversation:\n"
        )
        for message in messages:
            message.pretty_print()
        break

    answer_question()

    print(json.dumps(global_usage_aggregator.get_aggregated_usage(), indent=2))
