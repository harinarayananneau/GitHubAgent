import os
import ast
import json
from typing import TypedDict, List, Dict, Annotated
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
import anthropic_mcp_client as mcp
import asyncio

# Configuration
GITHUB_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
REPO_OWNER = "harinarayananneau"
REPO_NAME = "GitHubAgent"
TARGET_BRANCH = "main"

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=GEMINI_API_KEY,
    temperature=0.3
)

# State definition for LangGraph
class AgentState(TypedDict):
    repo_owner: str
    repo_name: str
    branch: str
    files_to_process: List[str]
    current_file: str
    file_content: str
    functions_without_docstrings: List[Dict]
    updated_content: str
    commit_message: str
    status: str
    error: str

# MCP Client for GitHub operations
class GitHubMCPClient:
    def __init__(self):
        self.client = None
    
    async def initialize(self):
        """Initialize MCP connection to GitHub server"""
        self.client = await mcp.create_client("github")
    
    async def get_file_content(self, owner: str, repo: str, path: str, branch: str):
        """Fetch file content from GitHub"""
        result = await self.client.call_tool(
            "get_file_contents",
            {
                "owner": owner,
                "repo": repo,
                "path": path,
                "branch": branch
            }
        )
        return result
    
    async def list_repository_files(self, owner: str, repo: str, branch: str):
        """List all Python files in repository"""
        result = await self.client.call_tool(
            "search_repositories",
            {
                "query": f"repo:{owner}/{repo} language:python extension:py"
            }
        )
        return result
    
    async def create_or_update_file(self, owner: str, repo: str, path: str, 
                                   content: str, message: str, branch: str):
        """Create or update file in GitHub"""
        result = await self.client.call_tool(
            "create_or_update_file",
            {
                "owner": owner,
                "repo": repo,
                "path": path,
                "content": content,
                "message": message,
                "branch": branch
            }
        )
        return result
    
    async def create_pull_request(self, owner: str, repo: str, title: str, 
                                 body: str, head: str, base: str):
        """Create a pull request"""
        result = await self.client.call_tool(
            "create_pull_request",
            {
                "owner": owner,
                "repo": repo,
                "title": title,
                "body": body,
                "head": head,
                "base": base
            }
        )
        return result

github_client = GitHubMCPClient()

# Helper function to parse Python code and find functions without docstrings
def find_functions_without_docstrings(code: str, filename: str) -> List[Dict]:
    """Parse Python code and identify functions missing docstrings"""
    functions_missing_docs = []
    
    try:
        tree = ast.parse(code)
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                docstring = ast.get_docstring(node)
                
                if not docstring:
                    # Get function signature
                    args = [arg.arg for arg in node.args.args]
                    
                    functions_missing_docs.append({
                        "name": node.name,
                        "line_number": node.lineno,
                        "args": args,
                        "is_async": isinstance(node, ast.AsyncFunctionDef),
                        "file": filename
                    })
    
    except SyntaxError as e:
        print(f"Syntax error in {filename}: {e}")
    
    return functions_missing_docs

# Helper function to generate docstring using Gemini
async def generate_docstring(function_info: Dict, code_context: str) -> str:
    """Generate docstring for a function using Gemini LLM"""
    
    prompt = f"""You are a Python documentation expert. Generate a comprehensive docstring for the following function.

Function Name: {function_info['name']}
Arguments: {', '.join(function_info['args'])}
Is Async: {function_info['is_async']}

Code Context:
{code_context}

Requirements:
1. Use Google-style docstring format
2. Include a brief description
3. Document all parameters with types
4. Document return value if applicable
5. Add usage example if helpful
6. Keep it concise but informative

Generate ONLY the docstring content (without triple quotes):"""

    messages = [
        SystemMessage(content="You are an expert Python developer specializing in documentation."),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    return response.content.strip()

# Helper function to insert docstring into code
def insert_docstring(code: str, function_name: str, line_number: int, docstring: str) -> str:
    """Insert generated docstring into the code"""
    lines = code.split('\n')
    
    # Find the function definition line
    func_line_idx = line_number - 1
    
    # Find the indentation of the function body
    indent = ""
    for i in range(func_line_idx + 1, len(lines)):
        line = lines[i]
        if line.strip():
            indent = line[:len(line) - len(line.lstrip())]
            break
    
    # Format docstring with proper indentation
    docstring_lines = [f'{indent}"""']
    docstring_lines.extend([f'{indent}{line}' for line in docstring.split('\n')])
    docstring_lines.append(f'{indent}"""')
    
    # Insert after function definition
    lines.insert(func_line_idx + 1, '\n'.join(docstring_lines))
    
    return '\n'.join(lines)

# LangGraph Node Functions

async def fetch_repository_files(state: AgentState) -> AgentState:
    """Node 1: Fetch all Python files from the repository"""
    print(f"Fetching files from {state['repo_owner']}/{state['repo_name']} on branch {state['branch']}")
    
    try:
        await github_client.initialize()
        
        # List all Python files
        files = await github_client.list_repository_files(
            state['repo_owner'],
            state['repo_name'],
            state['branch']
        )
        
        # Filter for .py files
        python_files = [f for f in files if f.endswith('.py')]
        
        state['files_to_process'] = python_files
        state['status'] = "files_fetched"
        print(f"Found {len(python_files)} Python files")
        
    except Exception as e:
        state['error'] = str(e)
        state['status'] = "error"
    
    return state

async def analyze_file(state: AgentState) -> AgentState:
    """Node 2: Analyze current file for missing docstrings"""
    
    if not state['files_to_process']:
        state['status'] = "complete"
        return state
    
    # Get next file to process
    current_file = state['files_to_process'].pop(0)
    state['current_file'] = current_file
    
    print(f"Analyzing file: {current_file}")
    
    try:
        # Fetch file content
        file_data = await github_client.get_file_content(
            state['repo_owner'],
            state['repo_name'],
            current_file,
            state['branch']
        )
        
        state['file_content'] = file_data['content']
        
        # Find functions without docstrings
        functions = find_functions_without_docstrings(
            state['file_content'],
            current_file
        )
        
        state['functions_without_docstrings'] = functions
        
        if functions:
            print(f"Found {len(functions)} functions without docstrings")
            state['status'] = "needs_docstrings"
        else:
            print(f"All functions have docstrings")
            state['status'] = "files_fetched"  # Move to next file
        
    except Exception as e:
        state['error'] = str(e)
        state['status'] = "error"
    
    return state

async def generate_and_add_docstrings(state: AgentState) -> AgentState:
    """Node 3: Generate docstrings using Gemini and add them to code"""
    
    print(f"Generating docstrings for {len(state['functions_without_docstrings'])} functions")
    
    updated_code = state['file_content']
    
    try:
        for func_info in state['functions_without_docstrings']:
            print(f"  - Generating docstring for: {func_info['name']}")
            
            # Generate docstring using Gemini
            docstring = await generate_docstring(func_info, updated_code)
            
            # Insert docstring into code
            updated_code = insert_docstring(
                updated_code,
                func_info['name'],
                func_info['line_number'],
                docstring
            )
        
        state['updated_content'] = updated_code
        state['status'] = "docstrings_generated"
        
    except Exception as e:
        state['error'] = str(e)
        state['status'] = "error"
    
    return state

async def commit_changes(state: AgentState) -> AgentState:
    """Node 4: Commit updated file back to repository"""
    
    print(f"Committing changes to {state['current_file']}")
    
    try:
        commit_message = f"docs: Add docstrings to functions in {state['current_file']}"
        
        # Create new branch for changes
        new_branch = f"docs/add-docstrings-{state['current_file'].replace('/', '-')}"
        
        await github_client.create_or_update_file(
            state['repo_owner'],
            state['repo_name'],
            state['current_file'],
            state['updated_content'],
            commit_message,
            new_branch
        )
        
        # Create pull request
        await github_client.create_pull_request(
            state['repo_owner'],
            state['repo_name'],
            f"Add docstrings to {state['current_file']}",
            f"Automatically generated docstrings for functions in {state['current_file']}",
            new_branch,
            state['branch']
        )
        
        print(f"Created PR for {state['current_file']}")
        
        state['status'] = "files_fetched"  # Continue with next file
        
    except Exception as e:
        state['error'] = str(e)
        state['status'] = "error"
    
    return state

# Build LangGraph workflow
def create_docstring_agent():
    """Create the LangGraph agent workflow"""
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("fetch_files", fetch_repository_files)
    workflow.add_node("analyze_file", analyze_file)
    workflow.add_node("generate_docstrings", generate_and_add_docstrings)
    workflow.add_node("commit_changes", commit_changes)
    
    # Define edges
    workflow.set_entry_point("fetch_files")
    
    workflow.add_conditional_edges(
        "fetch_files",
        lambda state: state['status'],
        {
            "files_fetched": "analyze_file",
            "error": END
        }
    )
    
    workflow.add_conditional_edges(
        "analyze_file",
        lambda state: state['status'],
        {
            "needs_docstrings": "generate_docstrings",
            "files_fetched": "analyze_file",  # Process next file
            "complete": END,
            "error": END
        }
    )
    
    workflow.add_edge("generate_docstrings", "commit_changes")
    workflow.add_edge("commit_changes", "analyze_file")
    
    return workflow.compile()

# Main execution function
async def run_docstring_agent(repo_owner: str, repo_name: str, branch: str = "main"):
    """Run the docstring generation agent"""
    
    print(f"\n{'='*60}")
    print(f"Starting Docstring Generation Agent")
    print(f"Repository: {repo_owner}/{repo_name}")
    print(f"Branch: {branch}")
    print(f"{'='*60}\n")
    
    # Initialize state
    initial_state = {
        "repo_owner": repo_owner,
        "repo_name": repo_name,
        "branch": branch,
        "files_to_process": [],
        "current_file": "",
        "file_content": "",
        "functions_without_docstrings": [],
        "updated_content": "",
        "commit_message": "",
        "status": "initialized",
        "error": ""
    }
    
    # Create and run agent
    agent = create_docstring_agent()
    
    result = await agent.ainvoke(initial_state)
    
    print(f"\n{'='*60}")
    print(f"Agent Execution Complete")
    print(f"Final Status: {result['status']}")
    if result.get('error'):
        print(f"Error: {result['error']}")
    print(f"{'='*60}\n")
    
    return result

# CLI entry point
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python docstring_agent.py <repo_owner> <repo_name> [branch]")
        sys.exit(1)
    
    repo_owner = sys.argv[1]
    repo_name = sys.argv[2]
    branch = sys.argv[3] if len(sys.argv) > 3 else "main"
    
    asyncio.run(run_docstring_agent(repo_owner, repo_name, branch))