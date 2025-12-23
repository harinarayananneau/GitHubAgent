"""
Simple example showing how to connect to GitHub MCP server using the official Python SDK
"""
import os
import sys
import shutil
import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Get GitHub token from environment
GITHUB_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")

async def simple_github_mcp_example():
    """Basic example of using MCP to interact with GitHub"""
    
    print("🚀 Connecting to GitHub MCP Server...\n")
    
    # Configure the GitHub MCP server
    server_params = StdioServerParameters(
        command="npx",  # Use npx to run the server
        args=["-y", "@modelcontextprotocol/server-github"],  # GitHub MCP server package
        env={"GITHUB_PERSONAL_ACCESS_TOKEN": GITHUB_TOKEN}
    )
    
    # Connect to the MCP server using stdio transport
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            print("✅ Connected to GitHub MCP Server\n")
            
            # 1. List available tools
            print("📋 Available Tools:")
            tools_response = await session.list_tools()
            for tool in tools_response.tools:
                print(f"  - {tool.name}: {tool.description}")
            print()
            
            # 2. Example: Search for repositories
            print("🔍 Searching for repositories...")
            result = await session.call_tool(
                "search_repositories",
                arguments={
                    "query": "language:python stars:>1000",
                    "page": 1,
                    "perPage": 5
                }
            )
            
            # Parse the result
            if result.content and len(result.content) > 0:
                data = json.loads(result.content[0].text)
                print(f"Found {len(data.get('repositories', []))} repositories:\n")
                for repo in data.get('repositories', [])[:3]:
                    print(f"  ⭐ {repo['full_name']} - {repo['stargazers_count']} stars")
                    print(f"     {repo['description'][:80]}...")
                    print()
            
            # 3. Example: Get file contents from a repository
            print("📄 Getting file from repository...")
            try:
                file_result = await session.call_tool(
                    "get_file_contents",
                    arguments={
                        "owner": "octocat",  # Example repository
                        "repo": "Hello-World",
                        "path": "README"
                    }
                )
                
                if file_result.content and len(file_result.content) > 0:
                    file_data = json.loads(file_result.content[0].text)
                    content = file_data.get('content', '')
                    print(f"File content preview:\n{content[:200]}...\n")
            except Exception as e:
                print(f"Note: {e}\n")
            
            # 4. Example: List your own repositories
            print("📚 Your repositories:")
            try:
                my_repos_result = await session.call_tool(
                    "list_repositories",
                    arguments={
                        "page": 1,
                        "perPage": 5
                    }
                )
                
                if my_repos_result.content and len(my_repos_result.content) > 0:
                    my_repos_data = json.loads(my_repos_result.content[0].text)
                    for repo in my_repos_data.get('repositories', [])[:5]:
                        print(f"  📁 {repo['name']} - {repo.get('description', 'No description')}")
                print()
            except Exception as e:
                print(f"Note: {e}\n")
            
            print("✅ MCP Session Complete!")

# Async example for listing repository contents
async def list_repo_files_example(owner: str, repo: str):
    """Example: List all files in a repository"""
    
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env={"GITHUB_PERSONAL_ACCESS_TOKEN": GITHUB_TOKEN}
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            print(f"\n📂 Files in {owner}/{repo}:")
            
            result = await session.call_tool(
                "list_repository_contents",
                arguments={
                    "owner": owner,
                    "repo": repo,
                    "path": ""  # Root directory
                }
            )
            
            if result.content and len(result.content) > 0:
                data = json.loads(result.content[0].text)
                for item in data.get('contents', []):
                    icon = "📁" if item['type'] == 'dir' else "📄"
                    print(f"  {icon} {item['name']}")

# Example: Create an issue
async def create_issue_example(owner: str, repo: str, title: str, body: str):
    """Example: Create a GitHub issue"""
    
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env={"GITHUB_PERSONAL_ACCESS_TOKEN": GITHUB_TOKEN}
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            print(f"\n📝 Creating issue in {owner}/{repo}...")
            
            result = await session.call_tool(
                "create_issue",
                arguments={
                    "owner": owner,
                    "repo": repo,
                    "title": title,
                    "body": body
                }
            )
            
            if result.content and len(result.content) > 0:
                data = json.loads(result.content[0].text)
                print(f"✅ Issue created: #{data.get('number')}")
                print(f"   URL: {data.get('html_url')}")

if __name__ == "__main__":
    # Check if GitHub token is set
    if not GITHUB_TOKEN:
        print("❌ Error: GITHUB_PERSONAL_ACCESS_TOKEN environment variable not set")
        print("\nPlease set it with:")
        print("  export GITHUB_PERSONAL_ACCESS_TOKEN='your_token_here'")
        exit(1)
    
    print("="*60)
    print("GitHub MCP Client Example")
    print("="*60)

    # Check that `npx` is available on PATH (used to start the MCP server)
    if shutil.which("npx") is None:
        print("\n❌ Error: 'npx' not found on PATH. This script starts the GitHub MCP server via `npx`.")
        print("Install Node.js (which includes npm/npx) and ensure `npx` is available in your shell.")
        print("Download: https://nodejs.org/ (LTS recommended)")
        print("After install, verify with: npx --version\n")
        sys.exit(1)

    # Run the main example
    asyncio.run(simple_github_mcp_example())
    
    # Uncomment to try other examples:
    # asyncio.run(list_repo_files_example("octocat", "Hello-World"))
    # asyncio.run(create_issue_example("your-username", "your-repo", 
    #                                   "Test Issue", "Created via MCP"))