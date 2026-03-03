FROM python:3.11

# Set the working directory in the container
RUN mkdir /tpmcp
WORKDIR /tpmcp
RUN mkdir ./llm
RUN mkdir ./rag
RUN mkdir ./rag/resources
RUN mkdir ./literature_search
RUN mkdir ./.config

# Copy the requirements file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY ./src/ ./

# Create a non-root user for security best practices
RUN useradd --create-home --shell /bin/bash tpmcp-user && chown -R tpmcp-user:tpmcp-user /tpmcp
USER tpmcp-user

# Expose the port your FastMCP server runs on (default is often dynamic, but 8080 is a common choice for HTTP deployment)
EXPOSE 9222

# Define the command to run your FastMCP server
CMD ["fastmcp", "run", "tpmcp.py", "--transport", "http", "--host", "0.0.0.0", "--port", "9222"]

