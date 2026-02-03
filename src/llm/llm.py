from langchain_openai import AzureChatOpenAI
from utils import Config

def create_llm_for_search():
    llm = AzureChatOpenAI(
        azure_endpoint=Config.env_config["AZURE_OPENAI_ENDPOINT"], # load from .env
        openai_api_key=Config.env_config["AZURE_OPENAI_API_KEY"], # load from .env
        model_name="azure-gpt-5",
        temperature=1,
        api_version=Config.env_config["OPENAI_API_VERSION"],
        max_retries=10,
        max_completion_tokens=4096,
        seed=42,
        tiktoken_model_name="gpt-4o", # use the gpt-4o tiktoken model for all models to avoid an error when calculating token limit
        http_client=Config.http_client
    )
    return llm