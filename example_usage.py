from agent_1 import DBAgent, DBAgentState

db_url = "postgres://postgres:123456@localhost:5432/hire10x"

agent = DBAgent(db_url=db_url)
qdrant_store = agent.create_memory()

print(qdrant_store.__dict__)

