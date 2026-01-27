import asyncio
from langchain_openai import AzureChatOpenAI
from mcp_use import MCPAgent, MCPClient
from dotenv import dotenv_values
import os
from pathlib import Path

DIR_HOME = Path(__file__).parent.parent

env_config = dotenv_values(DIR_HOME / ".config" / "example.env")
if os.path.exists(DIR_HOME / ".config" / ".env"):
    env_config = dotenv_values(DIR_HOME / ".config" / ".env")

AZURE_OPENAI_API_KEY = env_config["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_ENDPOINT = env_config["PUBLIC_AZURE_OPENAI_ENDPOINT"]
OPENAI_API_VERSION = env_config["OPENAI_API_VERSION"]

async def call_toxpipe_agent(model="azure-gpt-5", max_retries=10, max_tokens=9999, seed=42, reasoning_effort=None, http_client=None, query="", mcp_server_url="http://localhost:9222/mcp", system_prompt="", additional_instructions=""):

    model = "azure-gpt-5"
    max_retries = 10
    max_tokens = 9999
    seed = 42
    reasoning_effort = None
    http_client = None

    mcp_config = {
        "mcpServers": {
            "http": {
                "url": mcp_server_url
            }
        }
    }

    client = MCPClient.from_dict(mcp_config)

    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_key=AZURE_OPENAI_API_KEY,
        model_name=model,
        temperature=1,
        api_version=OPENAI_API_VERSION,
        max_retries=max_retries,
        max_completion_tokens=max_tokens,
        seed=seed,
        reasoning_effort=reasoning_effort,
        http_client=http_client
    )

    agent = MCPAgent(
        llm=llm,
        client=client,
        max_steps=20,
        system_prompt=system_prompt,
        additional_instructions=additional_instructions,
        verbose=True
        disallowed_tools=[] # may want to disable RAG or literature search depending on how you run tests
    )

    result = await agent.run(query)
    print(result, flush=True)
    return result

asyncio.run(call_toxpipe_agent())