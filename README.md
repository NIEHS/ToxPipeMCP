[![DOI](https://zenodo.org/badge/1129177352.svg)](https://doi.org/10.5281/zenodo.18852410)

# ToxPipeMCP
ToxPipeMCP is a [Model Context Protocol (MCP)](https://modelcontextprotocol.io/docs/getting-started/intro) server built using the [FastMCP](https://gofastmcp.com/getting-started/welcome) framework. This server functions as an API for AI Large-Language Models (LLMs) and provides these models with additional programmatic tools including:
- Access to the NIEHS's ChemBioTox database, a relational database with toxicological data form over 1 million chemicals, curated from reputable, scientific sources like the CTD, ToxRefDB, InVitroDB, DrugBank, and more
- PubMed literature search
- Search via retrieval augmented generation (RAG) from reports published by the NTP.

This MCP server is Dockerized and the [Dockerfile](https://github.com/NIEHS/ToxPipeMCP/blob/main/Dockerfile) is provided in the repository for quick and easy local setup.

To build, run the following command from the project's root directory:
`docker build . -t toxpipe/tpmcp`
