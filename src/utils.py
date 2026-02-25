from dotenv import dotenv_values
import os
from pathlib import Path

import ssl
import httpx
import truststore

truststore.inject_into_ssl()

class Config:

    # Import config file
    DIR_HOME = Path(__file__).parent
    env_config = dotenv_values(DIR_HOME / ".config" / "example.env")
    if os.path.exists(DIR_HOME / ".config" / ".env"):
        env_config = dotenv_values(DIR_HOME / ".config" / ".env")

    CERT_NAME = str(env_config["CERT_NAME"])

    http_client = None
    cert_path = DIR_HOME / ".config" / CERT_NAME

    if cert_path.exists():
        ctx = ssl.create_default_context(cafile=cert_path)  # Either cafile or capath.
        http_client = httpx.Client(verify=ctx)

    print(f'{http_client=}')