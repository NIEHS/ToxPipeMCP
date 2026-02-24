import asyncio
from vignettes import mcp_client

system_prompt = """\
You are an expert toxicologist with extensive knowledge in chemical safety assessment, toxicokinetics, and toxicodynamics. 
You will be given either a query from a user or an action from a previous thought. Analyze the query or action and perform the necessary action to proceed.
"""

query = "What chemicals are structurally similar to CC(C)(C1=CC=C(O)C=C1)C1=CC=C(O)C=C1"

response = asyncio.run(mcp_client.call_toxpipe_agent(
    model="azure-gpt-5",
    temperature=1,
    max_retries=10,
    max_tokens=9999,
    seed=42,
    reasoning_effort=None,
    http_client=None,
    query=query,
    mcp_server_url="http://localhost:9222/mcp",
    system_prompt=system_prompt,
    additional_instructions=""
))

print(response)