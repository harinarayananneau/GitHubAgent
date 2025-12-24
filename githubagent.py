import asyncio
import os
import sys
from contextlib import asynccontextmanager
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from dotenv import load_dotenv

load_dotenv()

@asynccontextmanager
async def github_mcp_client():
    """Create a GitHub MCP client with proper error handling"""
    github_token = os.environ.get('GITHUB_PERSONAL_ACCESS_TOKEN')
    
    if not github_token:
        raise ValueError("GITHUB_PERSONAL_ACCESS_TOKEN not found in environment")
    
    # Use docker with proper stdio handling
    server_params = StdioServerParameters(
        command="docker",
        args=[
            "run",
            "-i",
            "--rm",
            "-e", f"GITHUB_PERSONAL_ACCESS_TOKEN={github_token}",
            "ghcr.io/github/github-mcp-server:latest"
        ],
        env=None
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session

async def test_github_mcp():
    """Test connection to GitHub MCP Server running in Docker"""
    
    print("=" * 70)
    print("GitHub MCP Server Connection Test (stdio)")
    print("=" * 70)
    
    try:
        print("\n🚀 Starting GitHub MCP Server via Docker...")
        print("   (This may take a few seconds...)")
        
        async with github_mcp_client() as session:
            
            print("\n1️⃣  Initializing connection...")
            print("   ✅ Connection initialized successfully!")
            
            # List available tools
            print("\n2️⃣  Listing available tools...")
            try:
                tools_result = await asyncio.wait_for(session.list_tools(), timeout=10.0)
                tools = tools_result.tools
                print(f"   ✅ Found {len(tools)} tools:")
                for tool in tools[:10]:
                    desc = tool.description[:80] if tool.description else "No description"
                    print(f"      • {tool.name}: {desc}...")
                if len(tools) > 10:
                    print(f"      ... and {len(tools) - 10} more tools")
            except asyncio.TimeoutError:
                print("   ⚠️  Timeout while listing tools")
            
            # List available resources
            print("\n3️⃣  Listing available resources...")
            try:
                resources_result = await asyncio.wait_for(session.list_resources(), timeout=10.0)
                resources = resources_result.resources
                print(f"   ✅ Found {len(resources)} resources:")
                for resource in resources[:5]:
                    print(f"      • {resource.name}: {resource.uri}")
                if len(resources) > 5:
                    print(f"      ... and {len(resources) - 5} more resources")
            except asyncio.TimeoutError:
                print("   ⚠️  Timeout while listing resources")
            
            # List available prompts
            print("\n4️⃣  Listing available prompts...")
            try:
                prompts_result = await asyncio.wait_for(session.list_prompts(), timeout=10.0)
                prompts = prompts_result.prompts
                print(f"   ✅ Found {len(prompts)} prompts:")
                for prompt in prompts:
                    desc = prompt.description if prompt.description else "No description"
                    print(f"      • {prompt.name}: {desc}")
            except asyncio.TimeoutError:
                print("   ⚠️  Timeout while listing prompts")
            
            # Test a simple tool call
            print("\n5️⃣  Testing a tool call (get_file_contents)...")
            print("   (Fetching README from a public repo...)")
            try:
                result = await asyncio.wait_for(
                    session.call_tool("get_file_contents", {
                        "owner": "octocat",
                        "repo": "Hello-World",
                        "path": "README"
                    }),
                    timeout=15.0
                )
                print(f"   ✅ Tool call successful!")
                if result.content:
                    content_preview = str(result.content[0].text if hasattr(result.content[0], 'text') else result.content[0])[:200]
                    print(f"   📄 Response preview: {content_preview}...")
            except asyncio.TimeoutError:
                print(f"   ⚠️  Tool call timed out")
            except Exception as e:
                print(f"   ⚠️  Tool call failed: {e}")
            
            print("\n" + "=" * 70)
            print("✅ All tests completed!")
            print("=" * 70)
            
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("Set it with: $env:GITHUB_PERSONAL_ACCESS_TOKEN='your_token'")
    except FileNotFoundError:
        print("\n❌ Error: Docker not found. Please ensure Docker is installed and in PATH")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(f"\nDetails: {type(e).__name__}")
        
        # Check if it's the BrokenResourceError
        if "BrokenResourceError" in str(type(e).__name__) or "TaskGroup" in str(e):
            print("\n💡 Troubleshooting tips:")
            print("   1. Make sure Docker Desktop is running")
            print("   2. Try pulling the image first: docker pull ghcr.io/github/github-mcp-server:latest")
            print("   3. Verify your GitHub token is valid")
            print("   4. Check Docker logs for any errors")
        
        import traceback
        traceback.print_exc()

async def quick_test():
    """Quick test to verify basic connectivity"""
    print("Running quick connectivity test...\n")
    
    github_token = os.environ.get('GITHUB_PERSONAL_ACCESS_TOKEN')
    
    if not github_token:
        print("❌ GITHUB_PERSONAL_ACCESS_TOKEN not set!")
        return
    
    try:
        async with github_mcp_client() as session:
            print("✅ Connected successfully!")
            
            tools_result = await asyncio.wait_for(session.list_tools(), timeout=10.0)
            tools = tools_result.tools
            print(f"✅ Found {len(tools)} available tools\n")
            
            print("First 10 tools:")
            for i, tool in enumerate(tools[:10], 1):
                print(f"  {i}. {tool.name}")
                
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        import traceback
        traceback.print_exc()

async def interactive_test():
    """Interactive test to explore the MCP server"""
    print("=" * 70)
    print("GitHub MCP Server - Interactive Mode")
    print("=" * 70)
    
    try:
        async with github_mcp_client() as session:
            print("\n✅ Connected to GitHub MCP Server!")
            
            while True:
                print("\n" + "=" * 70)
                print("Options:")
                print("  1. List all tools")
                print("  2. List resources")
                print("  3. List prompts")
                print("  4. Call get_authenticated_user")
                print("  5. Exit")
                print("=" * 70)
                
                choice = input("\nEnter your choice (1-5): ").strip()
                
                if choice == "1":
                    tools_result = await session.list_tools()
                    tools = tools_result.tools
                    print(f"\n📋 Available Tools ({len(tools)}):")
                    for tool in tools:
                        print(f"\n  • {tool.name}")
                        if tool.description:
                            print(f"    {tool.description}")
                
                elif choice == "2":
                    resources_result = await session.list_resources()
                    resources = resources_result.resources
                    print(f"\n📦 Available Resources ({len(resources)}):")
                    for resource in resources:
                        print(f"  • {resource.name}: {resource.uri}")
                
                elif choice == "3":
                    prompts_result = await session.list_prompts()
                    prompts = prompts_result.prompts
                    print(f"\n💬 Available Prompts ({len(prompts)}):")
                    for prompt in prompts:
                        print(f"  • {prompt.name}")
                        if prompt.description:
                            print(f"    {prompt.description}")
                
                elif choice == "4":
                    print("\n🔍 Calling get_authenticated_user...")
                    result = await session.call_tool("get_authenticated_user", {})
                    print(f"\n✅ Success!")
                    for content in result.content:
                        print(f"\n{content}")
                
                elif choice == "5":
                    print("\n👋 Goodbye!")
                    break
                
                else:
                    print("\n❌ Invalid choice. Please enter 1-5.")
                    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    
    print("GitHub MCP Server Test Client\n")
    
    # Choose which test to run based on command line argument
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "quick":
            asyncio.run(quick_test())
        elif mode == "interactive":
            asyncio.run(interactive_test())
        else:
            print(f"Unknown mode: {mode}")
            print("Usage: python githubagent.py [quick|interactive]")
    else:
        # Default: run full test
        print("Running full test...\n")
        asyncio.run(test_github_mcp())