THE FOLLOWING ARE INSTRUCTIONS FOR YOU TO COMPLY AND FOLLOW WHEN BUILDING AN MCP SERVER WITH FASTMCP PYTHON

1. Introduction to FastMCP for PythonFastMCP is a high-level, Pythonic framework designed to simplify the creation of both Model Context Protocol (MCP) servers and clients. It handles the low-level protocol details, server management, and boilerplate code, allowing developers to focus on defining the tools, resources, and prompts that extend the capabilities of Large Language Models (LLMs).While the official MCP SDKs provide foundational tools, FastMCP offers a more complete, production-focused toolkit that includes simplified APIs, client utilities, authentication, server composition, and integrations with major AI platforms.Why Use FastMCP?Fast & Simple: A high-level, decorator-based API means less code and faster development. For most cases, decorating a function is all you need.Pythonic: The framework is designed to feel natural and intuitive for Python developers.Complete Ecosystem: FastMCP provides a comprehensive platform for the entire MCP lifecycle, from development and testing to deployment and integration.2. Building a FastMCP Server: Core ConceptsThe central component of any FastMCP application is the FastMCP server class. This class acts as a container for all your server's capabilities.from fastmcp import FastMCP

# Initialize a server instance
mcp = FastMCP(name="MyAwesomeServer", version="1.0.0")

# The server can be run from the main block
if __name__ == "__main__":
    mcp.run() # Defaults to 'stdio' transport
a. Tools: Exposing Executable FunctionsTools are the most common primitive, allowing an LLM to perform actions. In FastMCP, you create a tool by decorating a Python function.Decorator: @mcp.toolHow it works: FastMCP inspects the function's signature (name, docstring, type hints) to automatically generate the tool's name, description, and input schema.from fastmcp import FastMCP
from typing import List

mcp = FastMCP(name="Calculator")

@mcp.tool
def add(a: int, b: int) -> int:
    """Adds two integer numbers together."""
    return a + b

@mcp.tool
def sum_list(numbers: List[float]) -> float:
    """Calculates the sum of a list of numbers."""
    return sum(numbers)
Key Tool Features:Parameter Validation: FastMCP uses Pydantic under the hood for robust type validation and coercion. You can use Annotated and Field for advanced validation (e.g., ge, le, pattern).Return Values: Your function can return various types, which FastMCP automatically converts to the correct MCP content format (e.g., str -> TextContent, dict -> JSON string, Image -> ImageContent).Error Handling: Raise a ToolError to send a user-facing error message to the client. Other exceptions are logged, and their details can be masked for security (mask_error_details=True).Annotations: Provide hints to the client UI about a tool's behavior (e.g., readOnlyHint, destructiveHint) via the annotations argument in the decorator.b. Resources: Exposing DataResources provide read-only access to data, like files or database content.Decorator: @mcp.resource("unique-uri")How it works: You decorate a function that returns the content of the resource. The function is only executed when a client requests that specific URI.from fastmcp import FastMCP

mcp = FastMCP(name="DataServer")

@mcp.resource("resource://app/config")
def get_app_config() -> dict:
    """Returns the application's configuration as JSON."""
    return {"version": "1.0", "feature_flags": ["new_ui"]}
Resource Templates:For dynamic resources, use URI placeholders that map to function arguments.@mcp.resource("users://{user_id}/profile")
def get_user_profile(user_id: int) -> dict:
    """Retrieves a user's profile by their ID."""
    # In a real app, you'd fetch this from a database
    return {"id": user_id, "name": f"User {user_id}", "status": "active"}
A client can now request users://123/profile to get the profile for user 123.c. Prompts: Reusable TemplatesPrompts are user-controlled templates for interacting with an LLM.Decorator: @mcp.promptHow it works: The decorated function takes arguments and returns a string or a list of messages that will be sent to the LLM.from fastmcp import FastMCP

mcp = FastMCP(name="PromptServer")

@mcp.prompt
def summarize_text(text: str, length: str = "short") -> str:
    """Generates a prompt to summarize the given text."""
    return f"Please provide a {length} summary of the following text: {text}"
3. Advanced Server Featuresa. Accessing the MCP ContextWithin any tool, resource, or prompt function, you can access MCP session capabilities by adding a parameter type-hinted as Context.from fastmcp import FastMCP, Context

mcp = FastMCP(name="ContextDemo")

@mcp.tool
async def process_data(data_id: str, ctx: Context) -> str:
    await ctx.info(f"Starting processing for data ID: {data_id}")
    # ... long-running task ...
    await ctx.report_progress(progress=50, total=100)
    # ... more work ...
    await ctx.report_progress(progress=100, total=100)
    await ctx.info("Processing complete.")
    return "Success"
Context provides access to:Logging: ctx.info(), ctx.warn(), ctx.error()Progress Reporting: ctx.report_progress()Resource Access: await ctx.read_resource(uri)LLM Sampling: await ctx.sample(...) to ask the client's LLM for a completion.b. Server Composition & Proxyingmount(): Dynamically link multiple FastMCP servers together. Requests to the main server are delegated to the appropriate sub-server at runtime. This is ideal for modular application design.as_proxy(): Create a FastMCP server that acts as a frontend for another MCP server. This is perfect for bridging transports (e.g., exposing a remote HTTP server locally via stdio) or adding a layer of logic (caching, auth) in front of an existing server.c. AuthenticationFastMCP provides robust, out-of-the-box authentication for servers using HTTP transports.Bearer Token Auth: Secure your server by validating JWT Bearer tokens using a public key or JWKS URI.OAuth 2.1: For user-facing applications, FastMCP includes a helper that simplifies the full Authorization Code Flow with PKCE.4. Running, Deploying, and Testinga. Running the ServerYou can run your server using the run() method or the FastMCP CLI.Transports:stdio (default): For local use with clients like Claude Desktop.streamable-http (recommended for web): Exposes the server over HTTP.sse (legacy): An older HTTP-based transport.# Run with the default stdio transport
python my_server.py

# Run as a web server using the CLI
fastmcp run my_server.py --transport streamable-http --port 8080
b. Integrating with ASGI ApplicationsYou can mount a FastMCP server within existing web frameworks like Starlette or FastAPI to add MCP capabilities to a larger application.c. TestingIn-Memory Testing: The most efficient way to test is to pass the server instance directly to a fastmcp.Client. This avoids network overhead and is ideal for unit tests with pytest.FastMCP CLI:fastmcp dev server.py: Runs your server with the MCP Inspector for interactive debugging.fastmcp install server.py: A powerful command that automatically configures and installs your server (with its dependencies) into Claude Desktop.5. Client-Side InteractionsWhile this guide focuses on servers, understanding the client is useful. The fastmcp.Client provides a high-level async interface to interact with any MCP server.Automatic Transport Inference: The client can automatically determine the correct transport based on the target (e.g., a file path for stdio, a URL for HTTP).Advanced Features: The client can be configured with handlers for server-side logging, progress updates, and sampling requests.Multi-Server Support: A single client can be configured to interact with multiple servers, with tool and resource names automatically prefixed to avoid conflicts.6. IntegrationsFastMCP servers can be seamlessly integrated with major AI platforms that support the MCP standard:Claude Desktop: The primary local host for MCP servers.Anthropic & OpenAI APIs: Both support calling tools from remote MCP servers deployed to a public URL.Google Gemini SDK: The Python SDK can interact directly with a FastMCP client session.This comprehensive support makes FastMCP a versatile choice for building tools that can be used across the AI ecosystem.

FastMCP: The Complete Developer's GuideThis guide provides a comprehensive, step-by-step walkthrough for developers to build, test, deploy, and interact with MCP servers and clients using the FastMCP framework. We will cover everything from basic setup to advanced patterns like authentication, server composition, and integration with AI platforms.Part 1: Getting Started and Core Concepts1.1. Environment SetupFirst, ensure you have Python 3.10+ installed. We'll use uv, a fast package installer, to manage our environment.Install uv:# On macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
irm https://astral.sh/uv/install.ps1 | iex
Create a Project Directory and Virtual Environment:mkdir fastmcp-project
cd fastmcp-project
uv venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
Install FastMCP:uv pip install "fastmcp[all]"
The [all] option includes dependencies for features like HTTP servers and OpenAPI integration.1.2. Building Your First ServerLet's create a server that exposes a few basic capabilities. Create a file named server.py.# server.py

import asyncio
from fastmcp import FastMCP, Context
from fastmcp.prompts import Message
from typing import Literal
from pydantic import Field
from typing_extensions import Annotated
import httpx # Make sure to install: uv pip install httpx

# 1. Initialize the FastMCP Server
# The `name` is for identification, and `instructions` can guide LLMs.
mcp = FastMCP(
    name="MyAwesomeServer",
    instructions="This server provides utility tools for calculations, data fetching, and content processing."
)

# 2. Define a Tool with @mcp.tool()
# NOTE: The parentheses are required, even with no arguments.
# Tools are functions that perform actions. FastMCP uses type hints
# and docstrings to generate a schema for the LLM.
@mcp.tool()
def add(a: int, b: int) -> int:
    """Adds two integer numbers together and returns the sum."""
    return a + b

@mcp.tool(
    name="get_weather",
    description="Fetches the current weather for a specific city.",
    tags={"api", "weather"}
)
async def fetch_weather_data(
    city: Annotated[str, Field(description="The city name, e.g., 'San Francisco'")],
    units: Annotated[Literal["metric", "imperial"], Field(description="The unit system for temperature.")] = "metric"
) -> dict:
    """Fetches weather data from an external API asynchronously."""
    async with httpx.AsyncClient() as client:
        try:
            # Using a public weather API for demonstration
            response = await client.get(f"https://api.open-meteo.com/v1/forecast?latitude=37.77&longitude=-122.42&current_weather=true")
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            return {"error": f"API request failed: {e}"}

# 3. Define a Resource with @mcp.resource()
# Resources provide read-only data.
@mcp.resource("data://config/app_version")
def get_app_version() -> str:
    """Provides the current application version."""
    return "1.0.0"

# 4. Define a Resource Template for dynamic data
# The URI contains placeholders that map to function arguments.
@mcp.resource("users://{user_id}/profile")
def get_user_profile(user_id: int) -> dict:
    """Retrieves a user's profile by their ID."""
    # In a real app, you'd fetch this from a database.
    if user_id == 1:
        return {"id": 1, "name": "Alice", "status": "active"}
    return {"error": "User not found"}

# 5. Define a Prompt for reusable LLM interactions
@mcp.prompt()
def summarize_text_prompt(text_to_summarize: str) -> list[Message]:
    """Generates a structured prompt to ask an LLM to summarize text."""
    return [
        Message(role="user", content=f"Please summarize the following text concisely:\n\n{text_to_summarize}"),
        Message(role="assistant", content="Of course. Here is the summary:")
    ]

# 6. Create a Tool that uses the MCP Context
# The `Context` object gives you access to logging, progress reporting,
# resource access, and LLM sampling.
@mcp.tool()
async def process_document(doc_uri: str, ctx: Context) -> str:
    """Processes a document from a resource URI with logging and progress."""
    await ctx.info(f"Starting processing for document: {doc_uri}")

    # Use the context to read another resource on the server
    content_list = await ctx.read_resource(doc_uri)
    if not content_list or not hasattr(content_list[0], 'text'):
        await ctx.error("Document not found or is empty.")
        return "Error: Document is empty."
    
    document_text = content_list[0].text
    await ctx.report_progress(progress=50, total=100, message="Document read.")

    # Use the context to ask the client's LLM to perform a task
    summary_response = await ctx.sample(f"Summarize this in one sentence: {document_text[:500]}")

    await ctx.report_progress(progress=100, total=100, message="Processing complete.")
    await ctx.info("Document processing finished.")
    
    return f"Document processed. Summary: {summary_response.text}"

# 7. Add a `run` block to make the server executable
if __name__ == "__main__":
    # The default transport is 'stdio', used by clients that run the server as a local process.
    # mcp.run(transport="stdio")

    # For web-based access, use 'streamable-http'.
    print("Running server on http://127.0.0.1:8000")
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8000)
Part 2: Testing and Development WorkflowRobust testing is crucial. FastMCP is designed to be highly testable.2.1. In-Memory Unit Testing with pytestThis is the most efficient way to test your server's logic.Install pytest:uv pip install pytest pytest-asyncio
Create a test file test_server.py:# test_server.py
import pytest
from fastmcp import Client
from server import mcp as mcp_server # Import your server instance

@pytest.fixture
def mcp():
    """Provides the server instance for tests."""
    return mcp_server

@pytest.mark.asyncio
async def test_add_tool(mcp):
    # Pass the server instance directly to the Client for in-memory testing.
    # This avoids network/process overhead and is extremely fast.
    async with Client(mcp) as client:
        result = await client.call_tool("add", {"a": 15, "b": 10})
        assert result[0].text == '25'

@pytest.mark.asyncio
async def test_get_user_profile_resource(mcp):
    async with Client(mcp) as client:
        # Test the resource template
        resource_content = await client.read_resource("users://1/profile")
        # The result is JSON, so we check for a substring.
        assert '"name": "Alice"' in resource_content[0].text

        # Test a case where the user is not found
        not_found_content = await client.read_resource("users://99/profile")
        assert '"error": "User not found"' in not_found_content[0].text
Run the tests:pytest
2.2. Interactive Development with the MCP InspectorFor interactive testing and debugging, use the fastmcp dev command. It starts your server and a graphical "Inspector" tool in your browser.# This command runs your server in an isolated environment and launches the Inspector.
# Note: It runs the server over STDIO, so the Inspector will connect to it that way.
fastmcp dev server.py
This is excellent for manually calling tools with different parameters and inspecting the results.2.3. Organizing Code with Classes (Important Pattern)As your application grows, you'll want to use classes. Do not apply FastMCP decorators directly to instance or class methods. This will fail because the decorator captures the method before it's bound to an instance (self).The Correct Pattern: Register the bound method after creating an instance.# refactored_server.py
from fastmcp import FastMCP

class MyToolbox:
    def __init__(self, multiplier: int):
        self.multiplier = multiplier

    def multiply(self, a: int, b: int) -> int:
        """Multiplies two numbers and applies a configured multiplier."""
        return (a * b) * self.multiplier

# --- In your main server setup ---
mcp = FastMCP("ClassBasedServer")

# 1. Create an instance of your class
toolbox_instance = MyToolbox(multiplier=2)

# 2. Register the BOUND method with the server.
# `toolbox_instance.multiply` is a bound method where `self` is already set.
mcp.tool()(toolbox_instance.multiply) 
This pattern ensures that self is correctly handled and not exposed as a parameter to the LLM.Part 3: Building a Client ApplicationLet's create a client to interact with our running server.Start the Server: First, run server.py to get the HTTP server running.python server.py
Create client.py:# client.py
import asyncio
from fastmcp import Client

# Handlers for server-side events
async def handle_server_logs(log_message):
    print(f"[SERVER LOG | {log_message.level.upper()}]: {log_message.data}")

async def handle_progress(progress, total, message):
    print(f"[PROGRESS]: {int(progress)}/{int(total)} - {message}")

async def main():
    # Connect to the running HTTP server
    client = Client(
        "http://127.0.0.1:8000/mcp",
        log_handler=handle_server_logs,
        progress_handler=handle_progress
    )

    async with client:
        print("--- Calling 'add' tool ---")
        add_result = await client.call_tool("add", {"a": 100, "b": 23})
        print(f"Result: {add_result[0].text}")

        print("\n--- Reading 'user profile' resource ---")
        profile_res = await client.read_resource("users://1/profile")
        print(f"Resource Content: {profile_res[0].text}")

        print("\n--- Calling tool that uses Context ---")
        # This tool will trigger our log and progress handlers
        process_result = await client.call_tool(
            "process_document", 
            {"doc_uri": "data://config/app_version"}
        )
        print(f"Processing Result: {process_result[0].text}")

if __name__ == "__main__":
    asyncio.run(main())
Run the Client:python client.py
You will see the output from the tool calls, as well as the logs and progress updates sent from the server's Context object.Part 4: Authentication, Composition, and Proxying4.1. Securing Your ServerFor any public-facing server, you need authentication. We'll use Bearer Tokens (JWT).Update server.py to require authentication:# Add to server.py
from fastmcp.server.auth import BearerAuthProvider
from fastmcp.server.auth.providers.bearer import RSAKeyPair

# For testing, we can generate a key pair and a token.
# In production, you would use an external Identity Provider (IdP).
key_pair = RSAKeyPair.generate()
test_token = key_pair.create_token(audience="my-mcp-app")
print(f"\n--- AUTH TOKEN FOR TESTING ---\n{test_token}\n-----------------------------\n")

# Create an auth provider that validates tokens using the public key
auth_provider = BearerAuthProvider(
    public_key=key_pair.public_key,
    audience="my-mcp-app"
)

# Re-initialize the server with the auth provider
# NOTE: In a real app, you would only have one `mcp` instance.
# This is for demonstration.
secure_mcp = FastMCP(
    name="MyAwesomeServer",
    auth=auth_provider
)
# You would then re-register all your tools on `secure_mcp`.
# To keep this example simple, we'll omit re-registering tools here.
# In the main run block, you would then run `secure_mcp` instead of `mcp`.
Update client.py to authenticate:# Update client.py
# Paste the token printed by the server here
AUTH_TOKEN = "your-pasted-jwt-token"

client = Client(
    "http://127.0.0.1:8000/mcp",
    auth=AUTH_TOKEN, # Provide the token here
    # ... other handlers ...
)
Now, if the client tries to connect without a valid token, the server will reject the request with a 401 Unauthorized error.4.2. Server Composition and ProxyingComposition (mount): Build modular applications by mounting one server into another. The main server delegates requests to the sub-server.# main_server.py
from fastmcp import FastMCP
# Assume sub_mcp is defined in another file, sub_server.py
from sub_server import mcp as sub_mcp 

main_mcp = FastMCP("MainApp")
main_mcp.mount("utils", sub_mcp) # Mount sub_mcp under the "utils" prefix

# A tool named "do_stuff" in sub_mcp is now available as "utils_do_stuff"
# A resource "data://info" in sub_mcp is now "data://utils/info"
Proxying (as_proxy): Create a server that acts as a frontend for another (often remote) server. This is perfect for bridging transports.Use Case: Make a remote HTTP server accessible to a local client (like Claude Desktop) that only supports STDIO.# create_stdio_proxy.py
from fastmcp import FastMCP

# Create a proxy for a remote HTTP server.
proxy = FastMCP.as_proxy(
    "https://some-remote-api.com/mcp",
    name="RemoteAPIProxy"
)

if __name__ == "__main__":
    # Run the proxy with STDIO transport. A local client can now
    # run this script and interact with the remote server.
    proxy.run(transport="stdio")
This guide covers the fundamental components and patterns for developing with FastMCP. By understanding these concepts—servers, tools, resources, context, clients, testing, and deployment patterns—you are well-equipped to build powerful, maintainable, and secure applications for the AI ecosystem.


YOU WILL COMPLY WITH THE FASTMCP PYTHON PROTOCOL ABOVE

      
DO NOT CORRUPT MCP COMMUNICATIONS, WHICH ARE STDIO OR SSE. THERE SHOULD BE NO PRINT STATEMENTS AND ALL LOGGING SHOULD BE STDERR OR TO A SINGLE LOG FILE IN LOGS SUBDIRECTORY.
