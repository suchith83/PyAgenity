from agent_1 import DBAgent, DBAgentState

# db_url = "postgres://postgres:123456@localhost:5432/hire10x"

# agent = DBAgent(db_url=db_url)
# qdrant_store = agent.create_memory()

# print(qdrant_store.__dict__)

from pyagenity.store.mem0_store import Mem0Store, create_mem0_store
import os

from pyagenity.store.store_schema import MemoryType

from agent_1 import DBQueryAgent
from pyagenity.utils import Message

db_url = "postgres://postgres:123456@localhost:5432/hire10x"
config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "DB_AGENT_STORE",
            "url": os.getenv("QDRANT_URL"),
            "api_key": os.getenv("QDRANT_API_KEY"),
            "embedding_model_dims": 768,
        },
    },
    "embedder": {
        "provider": "gemini",
        "config": {"model": "models/text-embedding-004"},
    },
    "llm": {
        "provider": "gemini",
        "config": {"model": "gemini-2.0-flash-exp",
                    "temperature": 0.1},
    },

}
qdrant_store = create_mem0_store(config=config, user_id="db_agent", thread_id="db_agent_thread", app_id="db_agent")

agent = DBQueryAgent(db_url=db_url, db_store=qdrant_store)

app = agent._build_graph()

inp = {"messages": [Message.text_message("How many users are there in the user table?", role="user")]}
config = {"thread_id": "db_query_thread", "recursion_limit": 15}
result = app.invoke(inp, config=config)