import os

from dotenv import load_dotenv
from litellm import acompletion


import asyncpg

from pyagenity.graph import StateGraph
from pyagenity.state.agent_state import AgentState
from pyagenity.store.mem0_store import create_mem0_store, Mem0Store
from pyagenity.store.base_store import BaseStore
from pyagenity.store.store_schema import MemoryType
from pyagenity.utils import Message
from pyagenity.utils.constants import END
from pyagenity.utils.converter import convert_messages
from pyagenity.adapters.llm.model_response_converter import ModelResponseConverter

from pydantic import BaseModel
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional, List

from injectq import InjectQ, Inject

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

FACT_EXTRACTION_PROMPT = """
You are a Database Schema Fact Extractor. Your job is to read raw schema descriptions (DDL, information_schema output, or prose) and extract atomic, canonical facts about tables, columns, keys, and indexes.

Output format (must be JSON only):
{
    "facts": ["<fact1>", "<fact2>", ...]
}

Canonical fact templates (always use lowercase for identifiers, no quotes):
- "table {schema}.{table} exists"
- "description {schema}.{table}: {free_text_description}"
- "column {schema}.{table}.{column} type={data_type} nullable={true|false} default={value|none}"
- "primary_key {schema}.{table} ({col1,col2,...})"
- "foreign_key {schema}.{table}.{column} -> {ref_schema}.{ref_table}({ref_column})"
- "index {index_name} on {schema}.{table}({col1,col2,...}) unique={true|false}"

Normalization rules:
- Use lowercase for schema/table/column/index names.
- Use dot notation: schema.table and schema.table.column.
- For default values: write the literal if present, else "none".
- Keep one atomic assertion per fact string (do not bundle multiple columns or keys in one fact).
- If the input is irrelevant to schema, return {"facts": []}.

Examples:

Input:
Table: Public.Users
Columns:
 - id UUID PRIMARY KEY DEFAULT gen_random_uuid()
 - email TEXT UNIQUE NOT NULL
 - created_at TIMESTAMP DEFAULT now()
Indexes:
 - idx_users_email on email (unique)
Description: Application users

Output:
{
    "facts": [
        "table public.users exists",
        "description public.users: application users",
        "column public.users.id type=uuid nullable=false default=gen_random_uuid()",
        "primary_key public.users (id)",
        "column public.users.email type=text nullable=false default=none",
        "index idx_users_email on public.users(email) unique=true",
        "column public.users.created_at type=timestamp nullable=true default=now()"
    ]
}

Input:
Schema public.orders has column user_id referencing public.users(id).
Output:
{
    "facts": [
        "column public.orders.user_id type=unknown nullable=unknown default=none",
        "foreign_key public.orders.user_id -> public.users(id)"
    ]
}
"""

UPDATE_MEMORY_PROMPT = """
You are a smart database schema memory manager. You reconcile newly extracted schema facts with existing memory.

You can perform four operations per memory element:
- ADD: new fact not present
- UPDATE: same subject but details changed (e.g., type, nullability, default, key columns)
- DELETE: fact contradicts or subject removed (e.g., dropped column/index/table)
- NONE: duplicate or equivalent fact already present

Output format (must be JSON only):
{
    "memory": [
        {
            "id": "<existing_or_new_id>",
            "text": "<fact string>",
            "event": "ADD"|"UPDATE"|"DELETE"|"NONE",
            "old_memory": "<old fact string>"   # Required only for UPDATE
        },
        ...
    ]
}

Fact strings MUST use the canonical templates used during extraction:
- "table {schema}.{table} exists"
- "description {schema}.{table}: {free_text_description}"
- "column {schema}.{table}.{column} type={data_type} nullable={true|false} default={value|none}"
- "primary_key {schema}.{table} ({col1,col2,...})"
- "foreign_key {schema}.{table}.{column} -> {ref_schema}.{ref_table}({ref_column})"
- "index {index_name} on {schema}.{table}({col1,col2,...}) unique={true|false}"

Schema-specific guidance:
- New table/column/index/key → ADD
- Type change, nullability change, default change, index uniqueness/columns change, PK/FK columns change → UPDATE (keep same id, fill old_memory)
- Dropped table/column/index/key → DELETE
- Identical facts (string-equal after normalization) → NONE
- Column rename is not inferred; treat as DELETE (old) + ADD (new) unless both names appear with explicit "renamed" context
- Always use lowercase identifiers; dot-notation for paths; default=none if unspecified

Examples:

Old Memory:
[
    {"id":"1","text":"table public.users exists"},
    {"id":"2","text":"column public.users.email type=text nullable=false default=none"},
    {"id":"3","text":"index idx_users_email on public.users(email) unique=true"}
]

New facts:
[
    "table public.users exists",
    "column public.users.email type=text nullable=false default=none",
    "index idx_users_email on public.users(email) unique=false",
    "column public.users.created_at type=timestamp nullable=true default=now()"
]

Output:
{
    "memory": [
        {"id":"1","text":"table public.users exists","event":"NONE"},
        {"id":"2","text":"column public.users.email type=text nullable=false default=none","event":"NONE"},
        {
            "id":"3",
            "text":"index idx_users_email on public.users(email) unique=false",
            "event":"UPDATE",
            "old_memory":"index idx_users_email on public.users(email) unique=true"
        },
        {
            "id":"<new>",
            "text":"column public.users.created_at type=timestamp nullable=true default=now()",
            "event":"ADD"
        }
    ]
}
""" 
import logging

class ColorFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[1;31m' # Bold Red
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"

# --- global setup ---
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Clear default handlers added by basicConfig or libraries
root_logger.handlers.clear()

handler = logging.StreamHandler()
handler.setFormatter(ColorFormatter("%(levelname)-8s %(name)s - %(message)s"))
root_logger.addHandler(handler)

# Now every logger (yours + libraries) inherits this
logger = logging.getLogger(__name__)

# Example
logger.info("my info log")
logging.getLogger("some_library").warning("library warning")


class DBAgentState(AgentState):
    """ State helpful for storing memory in long term memory store """
    agent_id: str = "db_agent"
    db_url: str = ""
    schema_cache: Dict[str, Any] = Field(default_factory=dict)
    store_config: Dict[str, Any] = Field(default_factory=dict)
    

class SchemaInfo(BaseModel):
    table_name: str
    schema_name: str
    columns: list[dict[str, Any]]
    foreign_keys: list[dict[str, Any]]
    indexes: list[str]
    description: Optional[str] = "A database table schema"

class QueryTemplate(BaseModel):
    """Represents a SQL query template with placeholders."""
    name: str
    description: str
    template: str
    placeholders: Dict[str, str] = Field(default_factory=dict)  
    tags: List[str] = Field(default_factory=list)
    category: str
    

class DBAgent():
    """ Agent that stores information about db schema 
    in long term memory which can be queried later.
    """
    def __init__(
            self, 
            db_url: Optional[str] = None,
            mem0_config: Optional[Dict[str, Any]] = None,
    ):
        self.db_url = db_url or os.getenv("DATABASE_URL", "")
        self.config = mem0_config or {}
        self.default_templates = self._get_default_templates()

        # self._build_graph()

    def _get_default_templates(self) -> List[QueryTemplate]:
        """Get default SQL query templates."""
        return [
            QueryTemplate(
                name="table_info",
                description="Get information about tables in the database",
                template="""
                SELECT schemaname, tablename, tableowner 
                FROM pg_tables 
                WHERE schemaname NOT IN ('information_schema', 'pg_catalog')
                ORDER BY schemaname, tablename;
                """,
                placeholders={},
                tags=["schema", "tables", "info"],
                category="metadata"
            ),
            QueryTemplate(
                name="column_info",
                description="Get column information for a specific table",
                template="""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns 
                WHERE table_name = '{table_name}' 
                AND table_schema = '{schema_name}'
                ORDER BY ordinal_position;
                """,
                placeholders={
                    "table_name": "Name of the table to inspect",
                    "schema_name": "Schema name (usually 'public')"
                },
                tags=["columns", "schema", "table"],
                category="metadata"
            ),
            QueryTemplate(
                name="count_records",
                description="Count total records in a table",
                template="SELECT COUNT(*) as total_count FROM {schema_name}.{table_name};",
                placeholders={
                    "schema_name": "Schema name",
                    "table_name": "Table name"
                },
                tags=["count", "records"],
                category="analytics"
            ),
            QueryTemplate(
                name="recent_records",
                description="Get most recent records from a table with timestamp",
                template="""
                SELECT * FROM {schema_name}.{table_name} 
                ORDER BY {timestamp_column} DESC 
                LIMIT {limit};
                """,
                placeholders={
                    "schema_name": "Schema name",
                    "table_name": "Table name",
                    "timestamp_column": "Column containing timestamp/date",
                    "limit": "Number of records to return (default: 10)"
                },
                tags=["recent", "records", "limit"],
                category="data"
            ),
            QueryTemplate(
                name="search_records",
                description="Search for records containing a specific text value",
                template="""
                SELECT * FROM {schema_name}.{table_name} 
                WHERE {search_column} ILIKE '%{search_term}%'
                LIMIT {limit};
                """,
                placeholders={
                    "schema_name": "Schema name",
                    "table_name": "Table name",
                    "search_column": "Column to search in",
                    "search_term": "Text to search for",
                    "limit": "Number of records to return (default: 50)"
                },
                tags=["search", "text", "records"],
                category="data"
            )
        ]

    def _create_pg_pool(self, pg_pool: Any, postgres_dsn: str | None, pool_config: dict) -> Any:
        """
        Create or use an existing PostgreSQL connection pool.

        Args:
            pg_pool (Any, optional): Existing asyncpg Pool instance.
            postgres_dsn (str, optional): PostgreSQL connection string.
            pool_config (dict): Configuration for new pg pool creation.

        Returns:
            Pool: PostgreSQL connection pool.
        """
        if pg_pool:
            return pg_pool
        # as we are creating new pool, postgres_dsn must be provided
        # and we will release the resources if needed
        self.release_resources = True
        return asyncpg.create_pool(dsn=postgres_dsn, **pool_config)  # type: ignore

    async def _introspect_database_schema(self, state: DBAgentState) -> List[SchemaInfo]:
        """Introspect database to get schema information."""
        schema_info = []
        pg_pool = await self._create_pg_pool(pg_pool=None, postgres_dsn=state.db_url, pool_config={})
        try:
            async with pg_pool.acquire() as connection:
                # Get tables
                tables_query = """
                SELECT schemaname, tablename 
                FROM pg_tables 
                WHERE schemaname NOT IN ('information_schema', 'pg_catalog')
                ORDER BY schemaname, tablename;
                """
                tables = await connection.fetch(tables_query)
                
                for table_row in tables:
                    schema_name = table_row['schemaname']
                    table_name = table_row['tablename']
                    
                    # Get columns
                    columns_query = """
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns 
                    WHERE table_name = $1 AND table_schema = $2
                    ORDER BY ordinal_position;
                    """
                    columns_result = await connection.fetch(columns_query, table_name, schema_name)
                    columns = [
                        {
                            "name": row['column_name'],
                            "type": row['data_type'],
                            "nullable": row['is_nullable'] == 'YES',
                            "default": row['column_default']
                        }
                        for row in columns_result
                    ]
                    
                    # Get foreign keys
                    fk_query = """
                    SELECT
                        kcu.column_name,
                        ccu.table_schema AS foreign_table_schema,
                        ccu.table_name AS foreign_table_name,
                        ccu.column_name AS foreign_column_name 
                    FROM information_schema.table_constraints AS tc 
                    JOIN information_schema.key_column_usage AS kcu
                        ON tc.constraint_name = kcu.constraint_name
                        AND tc.table_schema = kcu.table_schema
                    JOIN information_schema.constraint_column_usage AS ccu
                        ON ccu.constraint_name = tc.constraint_name
                        AND ccu.table_schema = tc.table_schema
                    WHERE tc.constraint_type = 'FOREIGN KEY' 
                        AND tc.table_name = $1 
                        AND tc.table_schema = $2;
                    """
                    fk_result = await connection.fetch(fk_query, table_name, schema_name)
                    foreign_keys = [
                        {
                            "column": row['column_name'],
                            "references_table": f"{row['foreign_table_schema']}.{row['foreign_table_name']}",
                            "references_column": row['foreign_column_name']
                        }
                        for row in fk_result
                    ]
                    
                    # Get indexes
                    indexes_query = """
                    SELECT indexname
                    FROM pg_indexes
                    WHERE tablename = $1 AND schemaname = $2;
                    """
                    indexes_result = await connection.fetch(indexes_query, table_name, schema_name)
                    indexes = [row['indexname'] for row in indexes_result]
                    
                    schema_info.append(SchemaInfo(
                        table_name=table_name,
                        schema_name=schema_name,
                        columns=columns,
                        foreign_keys=foreign_keys,
                        indexes=indexes
                    ))
        
        except Exception as e:
            raise
        
        return schema_info

    async def store_schema(self, state: DBAgentState, store: Mem0Store):
        
        try:
            schema_info = await self._introspect_database_schema(state)
            
            for table_info in schema_info:
                schema_content = f"""
                Table: {table_info.schema_name}.{table_info.table_name}
                Description: {table_info.description or 'Database table'}
                Columns: {[f"{col['name']} ({col['type']})" for col in table_info.columns]}
                Column Details: {table_info.columns}
                Foreign Keys: {table_info.foreign_keys}
                Indexes: {table_info.indexes}
                """
                
                await store.astore(
                    config=state.store_config,
                    content=schema_content,
                    memory_type=MemoryType.SEMANTIC,
                    category="schema",
                    metadata={
                        "table_name": table_info.table_name,
                        "schema_name": table_info.schema_name,
                        "column_count": len(table_info.columns),
                        "has_foreign_keys": len(table_info.foreign_keys) > 0
                    },
                    infer=False
                )
            
            logger.info(f"Stored schema for {len(schema_info)} tables in memory")
        except Exception as e:
            logger.error(f"Failed to load and store schema: {e}")
    
    async def store_templates(self, state: DBAgentState, store: Mem0Store):
        """Store default query templates in memory."""

        try:
            for template in self.default_templates:
                template_content = f"""
                Template: {template.name}
                Description: {template.description}
                Category: {template.category}
                Tags: {', '.join(template.tags)}
                SQL Template: {template.template.strip()}
                Placeholders: {template.placeholders}
                """
                
                await store.astore(
                    config=state.store_config,
                    content=template_content,
                    memory_type=MemoryType.SEMANTIC,
                    category="template",
                    metadata={
                        "template_name": template.name,
                        "template_category": template.category,
                        "template_tags": template.tags,
                        "placeholders": template.placeholders
                    },
                    infer=False
                )
            
            logger.info(f"Stored {len(self.default_templates)} default templates in memory")
        except Exception as e:
            logger.error(f"Failed to store default templates: {e}")

    async def store_memory(self, state: DBAgentState, store: BaseStore | None = Inject[BaseStore]):
        """Store database schema and default templates in memory."""
        await self.store_schema(state, store)
        await self.store_templates(state, store)
        return state

    def _build_graph(self):
        default_config = {
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
            "custom_fact_extraction_prompt": FACT_EXTRACTION_PROMPT,
            "custom_update_memory_prompt": UPDATE_MEMORY_PROMPT,

        }
        if self.config:
            config = self.config
        else:
            config = default_config
        qdrant_store = create_mem0_store(config=config, user_id="db_agent", thread_id="db_agent_thread", app_id="db_agent_app")
        # store this object in InjectQ for easy access in states
        graph = StateGraph[DBAgentState](DBAgentState())
        graph.add_node("store_memory", self.store_memory)
        graph.add_edge("store_memory", END)
        graph.set_entry_point("store_memory")
        app =  graph.compile(store=qdrant_store)
        return app, qdrant_store
    
    def create_memory(self):
        app, qdrant_store = self._build_graph()
        input = {
            "messages": [
                Message.text_message(
                    "Store the database schema afnd default templates in memory",
                    role="user",
                )
            ],
            "state": {
                "db_url": self.db_url,
                "store_config": {"app_id": "db_agent", "thread_id": "db_agent_thread", "user_id": "db_agent"},
            },
        }
        config = {"thread_id": "db_agent_thread", "recursion_limit": 15}
        app.invoke(input, config=config)
        return qdrant_store
    
logging.basicConfig(level=logging.INFO)
class DBQueryState(AgentState):
    user_id: str = ""

class DBQueryAgent():
    """
    when we create longterm memory using DBAgent. And finally get the store object.
    We can use this store object to create DBQueryAgent.
    Which can use the stored schema and templates to answer questions.
    By generating sql queries using the templates and executing them on the database.

    - This also stores the previous queries and results in short term memory and also long term memory.
    - This uses those own memories to answer questions also.
    - This can also use the schema and templates stored in long term memory to generate better queries.
    """
    def __init__(
        self,
        db_store: Mem0Store,
        db_url: str = "",
    ):
        self.db_store = db_store
        self.db_url = db_url


    async def main_agent(self, state: DBQueryState, store: Any):
        # fetches the relevant schema and templates from long term memory (db_store)
        # Also fetches the relevant previous queries and results from own store.
        # Then uses all this information to generate a sql query using the templates.
        # Then executes the query on the database and gets the results.
        # Then uses the results to answer the user question. or Directly returns the results.
        messages = convert_messages([], state)
        user_message = messages[-1].content if messages else ""
        user_id = state.user_id
        memory_context = ""
        try:
            config = {"user_id": "db_agent", "thread_id": "db_agent_thread", "app_id": "db_agent"}
            memory_results = await self.db_store.asearch(
                config,
                query=user_message,
                limit=3,
                score_threshold=0.5,
                memory_type=MemoryType.SEMANTIC
            )
            if memory_results:
                memories = [result.content for result in memory_results]
                memory_context = f"\nRelevant memories from past conversations:\n" + "\n".join(
                    [f"- {memory}" for memory in memories]
                )
                print(f"📚 Retrieved {len(memories)} relevant memories")
            else:
                print("📚 No relevant memories found")
        except Exception as e:
            logger.error(f"Error in main_agent: {e}")

        system_prompt = f"""You are a helpful AI assistant with memory of past conversations.

{memory_context}

Be conversational, helpful, and reference past interactions when relevant. 
Show that you remember previous topics and user preferences."""

        # Convert messages for LLM
        messages = convert_messages(
            system_prompts=[{"role": "system", "content": system_prompt}], state=state
        )

        # Generate response using LiteLLM
        # try:
        response = await acompletion(
            model="gemini/gemini-2.0-flash-exp", messages=messages, temperature=0.7
        )

        assistant_content = response.choices[0].message.content
        assistant_message = Message.text_message(assistant_content, role="assistant")

        # Store the conversation in memory using store's message storage
        try:
            config = {"user_id": user_id, "thread_id": f"chat_{user_id}"}

            # Store user message
            user_msg = Message.text_message(user_message, role="user")
            await store.astore(
                config,
                content=user_msg,
                memory_type=MemoryType.EPISODIC,  # Use episodic for compatibility
                category="chat",
                metadata={"session_id": "main_chat", "interaction_type": "user_input"},
            )

            # Store assistant response
            assistant_msg = Message.text_message(assistant_content, role="assistant")
            await store.astore(
                config,
                content=assistant_msg,
                memory_type=MemoryType.EPISODIC,  # Use episodic for compatibility
                category="chat",
                metadata={
                    "session_id": "main_chat",
                    "interaction_type": "assistant_response",
                },
            )

            print(f"💾 Stored conversation for user {user_id}")

        except Exception as e:
            print(f"❌ Memory storage error: {e}")
            # Continue even if storage fails

        # Return updated state with new message
        return DBQueryState(context=[*state.context, assistant_message], user_id=state.user_id)