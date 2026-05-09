import os
import sys
from pathlib import Path

import requests
from langchain.messages import HumanMessage
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain.agents.middleware import ToolRetryMiddleware
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# import src.config.bootstrap  # noqa: F401

load_dotenv(".env")

CHINOOK_URL = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
DEFAULT_DB_PATH = Path("Chinook.db")


def downlaod_db(db_path: Path):
    if db_path.exists():
        print("db exists.")
        return

    response = requests.get(CHINOOK_URL)
    response.raise_for_status()
    db_path.write_bytes(response.content)

    print("database downloaded. ")


model = init_chat_model(model="ollama:gemma4:e2b")

# res = model.invoke("hello friend!")
# print(res.pretty_print())
sql_database = SQLDatabase.from_uri(f"sqlite:///{DEFAULT_DB_PATH}")

sql_toolkit = SQLDatabaseToolkit(llm=model, db=sql_database)

tools = sql_toolkit.get_tools()

system_prompt = f"""
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {sql_database.dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {os.environ.get("SQL_AGENT_TOP_K", "5")} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.
"""

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt,
    middleware=[
        ToolRetryMiddleware(
            max_retries=3,
            backoff_factor=2.0,
            initial_delay=1.0,
        ),
    ],
)

messages = []


def ask_question():
    last_message = None
    for chunk in agent.stream(
        {"messages": messages},
        stream_mode="values",
    ):
        if "messages" in chunk:
            last_message = chunk["messages"][-1]
            last_message.pretty_print()

    messages.append(last_message)


if __name__ == "__main__":
    while 1:
        question = input("> ")
        if question.lower() in {"exit", "quit"}:
            print(
                "\n\n\nConversation ended. Here are all the messages in the conversation:\n"
            )
            for message in messages:
                message.pretty_print()
            break

        # messages.append({"role": "user", "content": question})
        messages.append(HumanMessage(content=question))
        ask_question()
