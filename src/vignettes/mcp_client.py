import asyncio
import sys
from pathlib import Path
from urllib.parse import urlparse

from langchain_openai import AzureChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

try:
    from src.utils import Config
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from utils import Config

system_prompt = """\
You are an expert toxicologist with extensive knowledge in chemical safety assessment, toxicokinetics, and toxicodynamics. 
You will be given either a query from a user or an action from a previous thought. Analyze the query or action and perform the necessary action to proceed.
"""


def _normalize_mcp_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.path.endswith("/mcp"):
        return url
    path = parsed.path.rstrip("/")
    return parsed._replace(path=f"{path}/mcp").geturl()


def _force_http(url: str) -> str:
    parsed = urlparse(url)
    return parsed._replace(scheme="http").geturl()


def _mcp_url_candidates(raw_url: str) -> list[str]:
    normalized = _normalize_mcp_url(raw_url)
    parsed = urlparse(normalized)

    candidates = [normalized]

    if parsed.scheme == "https":
        candidates.append(_force_http(normalized))

    if parsed.port is None:
        with_default_port = parsed._replace(netloc=f"{parsed.hostname}:9222").geturl()
        candidates.append(with_default_port)
        if parsed.scheme == "https":
            candidates.append(_force_http(with_default_port))

    unique_candidates = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)

    return unique_candidates

async def call_toxpipe_agent(model="azure-gpt-5", temperature=1, max_retries=10, max_tokens=9999, seed=42, reasoning_effort=None, http_client=None, query="", mcp_server_url="", system_prompt="", additional_instructions=""):


    # model - model name to use - for ToxPipe, must be a valid model name from LiteLLM
    # temperature - temperature to affect model determinism. Values closer to 0 produce less variation and values closer to 1 produce more
    # max_retries - max number of retries the model can make if a call fails
    # max_tokens - max number of output tokens that can be in the model's response. Set this higher if using a newer/bigger model and expecting a long response
    # seed - seed for consistency in model randomness
    # reasoning_effort - enable reasoning capabilities (low/high) for models that support them
    # http_client - custom httpx client to be used with langchain/langgraph. May need to define a custom client if running into SSL errors

    # Define LLM using LangChain's class - all ToxPipe models use the AzureChatOpenAI API regardless of the actual model provider
    selected_http_client = http_client or Config.http_client

    llm = AzureChatOpenAI(
        azure_endpoint=Config.env_config['AZURE_OPENAI_ENDPOINT'],
        openai_api_key=Config.env_config['AZURE_OPENAI_API_KEY'],
        model_name=model,
        temperature=temperature,
        api_version=Config.env_config['OPENAI_API_VERSION'],
        max_retries=max_retries,
        max_completion_tokens=max_tokens,
        seed=seed,
        reasoning_effort=reasoning_effort,
        http_client=selected_http_client
    )

    selected_mcp_url = mcp_server_url or Config.env_config['MCP_SERVER_URL']
    candidate_urls = _mcp_url_candidates(selected_mcp_url)

    tools = None
    last_error = None

    for candidate_url in candidate_urls:
        client = MultiServerMCPClient(
            {
                "ToxPipeMCPServers": {
                    "transport": "http",
                    "url": candidate_url,
                }
            }
        )

        try:
            tools = await client.get_tools()
            break
        except Exception as exc:
            last_error = exc

    if tools is None:
        raise RuntimeError(
            f"Unable to connect to MCP server. Tried URLs: {candidate_urls}. Last error: {last_error}"
        ) from last_error

    agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)
    
    result = await agent.ainvoke({"messages": [{'role': 'user', 'content': query}]})

    return result['messages'][-1].content

if __name__=='__main__':
    asyncio.run(call_toxpipe_agent(
        model="azure-gpt-5",
        temperature=1,
        max_retries=10,
        max_tokens=9999,
        seed=42,
        reasoning_effort=None,
        http_client=Config.http_client,
        query="What is the toxicity of Aspirin?",
        mcp_server_url="http://127.0.0.1:9222/mcp",
        system_prompt="",
        additional_instructions=""
    ))
