import asyncio
from langchain_openai import AzureChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from ..utils import Config

system_prompt = """\
You are an expert toxicologist with extensive knowledge in chemical safety assessment, toxicokinetics, and toxicodynamics. 
You will be given either a query from a user or an action from a previous thought. Analyze the query or action and perform the necessary action to proceed.
"""

async def call_toxpipe_agent(model="azure-gpt-5", temperature=1, max_retries=10, max_tokens=9999, seed=42, reasoning_effort=None, http_client=None, query="", mcp_server_url="http://localhost:9222/mcp", system_prompt="", additional_instructions=""):


    # model - model name to use - for ToxPipe, must be a valid model name from LiteLLM
    # temperature - temperature to affect model determinism. Values closer to 0 produce less variation and values closer to 1 produce more
    # max_retries - max number of retries the model can make if a call fails
    # max_tokens - max number of output tokens that can be in the model's response. Set this higher if using a newer/bigger model and expecting a long response
    # seed - seed for consistency in model randomness
    # reasoning_effort - enable reasoning capabilities (low/high) for models that support them
    # http_client - custom httpx client to be used with langchain/langgraph. May need to define a custom client if running into SSL errors

    # Define LLM using LangChain's class - all ToxPipe models use the AzureChatOpenAI API regardless of the actual model provider
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
        http_client=Config.http_client
    )

    # Use langchain's MCP Client
    client = MultiServerMCPClient(
        {
            "ToxPipeMCPServers": {
                "transport": "http",
                "url": Config.env_config['MCP_SERVER_URL'],
            }
        }
    )

    tools = await client.get_tools()
    agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)
    
    result = await agent.ainvoke({"messages": [{'role': 'user', 'content': query}]})

    return result['messages'][-1].content

if __name__=='__main__':
    asyncio.run(call_toxpipe_agent())