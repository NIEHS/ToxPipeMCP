from langchain_openai import AzureChatOpenAI
import os
# Load environment variables
from dotenv import dotenv_values
from pathlib import Path
DIR_HOME = Path(__file__).parent.parent
env_config = dotenv_values(DIR_HOME / ".config" / "example.env")
if os.path.exists(DIR_HOME / ".config" / ".env"):
    env_config = dotenv_values(DIR_HOME / ".config" / ".env")

def create_llm_for_search():
    llm = AzureChatOpenAI(
        azure_endpoint=env_config["AZURE_OPENAI_ENDPOINT"], # load from .env
        openai_api_key=env_config["AZURE_OPENAI_API_KEY"], # load from .env
        model_name="azure-gpt-5",
        temperature=1,
        api_version=env_config["OPENAI_API_VERSION"],
        max_retries=10,
        max_completion_tokens=4096,
        seed=42,
        tiktoken_model_name="gpt-4o" # use the gpt-4o tiktoken model for all models to avoid an error when calculating token limit
    )
    return llm