from pathlib import Path
from dotenv import dotenv_values
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langchain_core.prompts import ChatPromptTemplate

from operator import add
from typing import Annotated, List

from typing_extensions import TypedDict

import ssl
import httpx
import truststore
truststore.inject_into_ssl()

# ---------------------------------------------------------------------------
class Config:
    DIR_HOME = Path(__file__).parent.parent
    DIR_DATA = (DIR_HOME / 'rag' / 'resources')
    
    http_client = None
    cert_path = DIR_HOME / '.config' / 'NIH-FULL.pem'
    if cert_path.exists():
        ctx = ssl.create_default_context(cafile=str(cert_path))  # Either cafile or capath.
        http_client = httpx.Client(verify=ctx)

    env_config = dotenv_values(DIR_HOME / ".config" / ".env")
    
    TOKENS_PER_LLM_CALL = 5000
    MAX_KEYPHRASES = 10
    MAX_NUM_DOCS = 5
    SIMILARITY_THRESHOLD = 0.3
    RETRY_COUNTER = 2

# ---------------------------------------------------------------------------
class State(TypedDict):
    query: str
    next_action: str
    response: str
    keyphrases: List[str]
    resources: str
    steps: Annotated[List[str], add]

# ---------------------------------------------------------------------------
def setPrompt(system_prompt, human_prompt):

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    system_prompt
                ),
            ),
            (
                "human",
                (
                    human_prompt
                ),
            ),
        ]
    )
