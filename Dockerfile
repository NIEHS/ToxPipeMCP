FROM python:3.11

# Set the working directory in the container
RUN mkdir /cbtmcp
WORKDIR /cbtmcp

# Copy the requirements file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY ./src/* ./
RUN mkdir ./.config/
COPY ./src/.config/.env ./.config/.env

# Create a non-root user for security best practices
RUN useradd --create-home --shell /bin/bash cbtmcp-user && chown -R cbtmcp-user:cbtmcp-user /cbtmcp
USER cbtmcp-user

# Expose the port your FastMCP server runs on (default is often dynamic, but 8080 is a common choice for HTTP deployment)
EXPOSE 9222

# Define the command to run your FastMCP server
CMD ["fastmcp", "run", "cbtmcp.py", "--transport", "http", "--host", "0.0.0.0", "--port", "9222"]

