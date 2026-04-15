from langchain.chat_models import init_chat_model
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from src.config.constants import DEFAULT_LLM_MODEL
from src.utils.usage_aggregator_callback import UsageAggregatorCallback
import src.config.bootstrap


global_usage_aggregator = UsageAggregatorCallback("test_task")


# both are valid

# llm = ChatOllama(
#     model=DEFAULT_LLM_MODEL, temperature=0, callbacks=[global_usage_aggregator]
# )
# llm = ChatOpenAI(
#     model="gpt-4.1-nano", temperature=0, callbacks=[global_usage_aggregator]
# )

llm = init_chat_model(
    model=DEFAULT_LLM_MODEL,
    model_provider="ollama",
    temperature=0,
    configurable_fields="any",
    callbacks=[global_usage_aggregator],
)

response = llm.invoke("how are you?")

print(response)

print(
    "Aggregated Usage Metadata:",
    global_usage_aggregator.get_aggregated_usage(),
)
