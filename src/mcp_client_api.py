from langchain_openai import AzureChatOpenAI
from mcp_use import MCPAgent, MCPClient
from dotenv import dotenv_values
from pathlib import Path
from fastapi import FastAPI, Request, Response

import httpx
import truststore
truststore.inject_into_ssl()

SYSTEM_PROMPT_TEMPLATE = """
You are an expert toxicologist with extensive knowledge in chemical safety assessment, toxicokinetics, and toxicodynamics. Your expertise includes:

1. Interpreting chemical structures and properties
2. Analyzing toxicological data from various sources (e.g., in vitro, in vivo, and in silico studies)
3. Applying read-across and QSAR (Quantitative Structure-Activity Relationship) approaches
4. Understanding mechanisms of toxicity and adverse outcome pathways
5. Evaluating systemic availability based on ADME (Absorption, Distribution, Metabolism, Excretion) properties
6. Assessing potential health hazards and risks associated with chemical exposure

When providing toxicological evaluations:
- Use reliable scientific sources and databases (e.g., PubChem, ECHA, EPA, IARC)
- Consider both experimental data and predictive models
- Explain your reasoning and cite relevant studies or guidelines
- Acknowledge uncertainties and data gaps
- Provide a balanced assessment, considering both potential hazards and mitigating factors
- Use a weight-of-evidence approach when multiple data sources are available
- Classify toxicodynamic activity and systemic availability as high, medium, or low based on 
the available evidence and expert judgment
- When using read-across, clearly state the basis for the analogy and any limitations
- If you are asked to perform multiple tasks or are asked multiple questions, provide a final answer for each task.

Adhere to ethical standards in toxicology and maintain scientific objectivity in your assessments. Always include the source for any information you provide. You must always distinguish which components of your final answer were sourced from tools and which were sourced from your training data. If possible, include the source of the information pulled from your training data.
If the answer to the query exists in the previous messages, you may skip tool or search usage and provide the answer directly.
Always use your available tools and consult your training data when responding to the user's query.

You will be given either a query from a user or an action from a previous thought. Analyze the query or action and perform the necessary action to proceed. Always follow the rules below:

**Rules**
- If a user asks a question that is not related to toxicology, chemicals, or biological terms, you must respond with the following message and do not make any tool calls: "This question is not about toxicology, chemicals, or biological terms. Therefore, I cannot answer this question."
- Always include whatever information you were able to find in your final answer.
- If your tool calls do not return any relevant information, you do not need to call those tools again.
- You must specify which part of the answer was sourced from your training data. You must also include a warning that the data was generated from training data and may not be accurate or up to date.
- You must analyze the user's query and make relevant tool calls as necessary. If your memory or tool history fully answers the user's query, you may skip calling tools.
- If you find, at any time, that the most recent response sufficiently answers the user's query, you may stop calling additional tools.
"""

DIR_HOME = Path(__file__).parent

http_client = None
cert_path = DIR_HOME / ".config/NIH-FULL.pem"
if cert_path.exists():
    http_client = httpx.Client(verify=cert_path)

app = FastAPI(
    title="ToxPipe MCP Client API",
    description="An API for creating custom AI agents for performing chat completions with specialized tool access via MCP. Part of the ToxPipe ecosystem.",
    version="0.0.1"
)

# Import config file
env_config = dotenv_values(DIR_HOME / ".config" / "example.env")
if (DIR_HOME / ".config" / ".env").exists():
    env_config = dotenv_values(DIR_HOME / ".config" / ".env")

# Load environment variables
AZURE_OPENAI_API_KEY = env_config["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_ENDPOINT = env_config["AZURE_OPENAI_ENDPOINT"]
OPENAI_API_VERSION = env_config["OPENAI_API_VERSION"]

@app.get("/mcp/", tags=["mcp"])
async def query_mcp_agent(request: Request, response: Response, query: str, model: str, temperature: float = 0., max_retries: int=10, max_tokens: int=9999, seed: int=42, reasoning_effort: bool | None=None):

    # model - model name to use - for ToxPipe, must be a valid model name from LiteLLM
    # temperature - temperature to affect model determinism. Values closer to 0 produce less variation and values closer to 1 produce more
    # max_retries - max number of retries the model can make if a call fails
    # max_tokens - max number of output tokens that can be in the model's response. Set this higher if using a newer/bigger model and expecting a long response
    # seed - seed for consistency in model randomness
    # reasoning_effort - enable reasoning capabilities (low/high) for models that support them

    # config dict to be passed to MCPClient. This can be npx commands for a locally-hosted MCP server, or a url for a remotely-hosted server
    mcp_config = {
        "mcpServers": {
            "http": {
                "url": env_config['MCP_SERVER_URL']
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
        system_prompt=SYSTEM_PROMPT_TEMPLATE, # custom system prompt for agents
        verbose=True,
        disallowed_tools=[] # may want to disable RAG or literature search depending on how you run tests
    )

    try:
        result = await agent.run(query) # Run agent without streaming
        error = ''
    except Exception as e:
        print("Error performing search with mcp agent.")
        print(e)
        response.status_code = 400
        result = ''
        error = f"Error: query failed to run with message: {e}."
    
    return {"response": result, "error": error}