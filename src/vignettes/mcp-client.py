import asyncio
from langchain_openai import AzureChatOpenAI
from mcp_use import MCPAgent, MCPClient
from dotenv import dotenv_values
import os
from pathlib import Path

# Import config file
DIR_HOME = Path(__file__).parent.parent
env_config = dotenv_values(DIR_HOME / ".config" / "example.env")
if os.path.exists(DIR_HOME / ".config" / ".env"):
    env_config = dotenv_values(DIR_HOME / ".config" / ".env")

# Load environment variables
AZURE_OPENAI_API_KEY = env_config["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_ENDPOINT = env_config["PUBLIC_AZURE_OPENAI_ENDPOINT"]
OPENAI_API_VERSION = env_config["OPENAI_API_VERSION"]

async def call_toxpipe_agent(model="azure-gpt-5", temperature=1, max_retries=10, max_tokens=9999, seed=42, reasoning_effort=None, http_client=None, query="", mcp_server_url="http://localhost:9222/mcp", system_prompt="", additional_instructions=""):


    # model - model name to use - for ToxPipe, must be a valid model name from LiteLLM
    # temperature - temperature to affect model determinism. Values closer to 0 produce less variation and values closer to 1 produce more
    # max_retries - max number of retries the model can make if a call fails
    # max_tokens - max number of output tokens that can be in the model's response. Set this higher if using a newer/bigger model and expecting a long response
    # seed - seed for consistency in model randomness
    # reasoning_effort - enable reasoning capabilities (low/high) for models that support them
    # http_client - custom httpx client to be used with langchain/langgraph. May need to define a custom client if running into SSL errors

    # config dict to be passed to MCPClient. This can be npx commands for a locally-hosted MCP server, or a url for a remotely-hosted server
    mcp_config = {
        "mcpServers": {
            "http": {
                "url": mcp_server_url
            }
        }
    }

    # Create MCP client
    client = MCPClient.from_dict(mcp_config)

    # Define LLM using LangChain's class - all ToxPipe models use the AzureChatOpenAI API regardless of the actual model provider
    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_key=AZURE_OPENAI_API_KEY,
        model_name=model,
        temperature=temperature,
        api_version=OPENAI_API_VERSION,
        max_retries=max_retries,
        max_completion_tokens=max_tokens,
        seed=seed,
        reasoning_effort=reasoning_effort,
        http_client=http_client
    )

    # Define agent 
    agent = MCPAgent(
        llm=llm,
        client=client,
        max_steps=20, # Max deliberation steps the agent can take - increase this if agent is not converging
        system_prompt=system_prompt, # custom system prompt for agents
        additional_instructions=additional_instructions, # see above
        verbose=True
        disallowed_tools=[] # may want to disable RAG or literature search depending on how you run tests
    )

    result = await agent.run(query) # Run agent without streaming
    print(result, flush=True) # Flushing to print immediately
    return result

asyncio.run(call_toxpipe_agent())