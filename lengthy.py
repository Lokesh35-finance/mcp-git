import streamlit as st
from pathlib import Path
import json
import subprocess
from langchain_aws.chat_models import ChatBedrockConverse
from langchain_core.messages import HumanMessage
from time import sleep
import logging

# --- Page Setup ---
st.set_page_config(page_title="GitHub Repo Chat", layout="wide")
st.sidebar.title("Menu")
st.sidebar.markdown("### GIT-HUB")

# --- User File Loading ---
user_file = Path("vitech-git-users.txt")
user_list = ["Select user"]
USER_ACCOUNTS=[]
if user_file.exists():
    USER_ACCOUNTS += [line.strip() for line in user_file.read_text().splitlines() if line.strip()]
else:
    USER_ACCOUNTS += user_list

# --- Logging Setup ---
logging.basicConfig(
    filename="mcp_failures.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# --- Session State for Memory ---
for key in ["chat", "cli", "last_question", "user", "repo", "branch"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "chat" else ""

# --- Prompt Function ---
def github_mcp_command(question: str) -> str:
    from github_prompt import base_prompt
    llm = ChatBedrockConverse(
        model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",
        region_name="us-east-1",
        performance_config={"latency": "optimized"},
        max_tokens=8192,
        temperature=0.1,
        top_p=1,
        stop_sequences=["\n\nHuman"]
    )
    full_prompt = f"{base_prompt}\n\n{question}"
    response = llm.invoke([HumanMessage(content=full_prompt)])
    return response.content

# --- Response Explanation Function ---
def explain_mcp_output(raw_response: str, original_question: str) -> str:
    prompt = f"""
You are a senior DevOps and GitHub CI/CD engineer. Interpret the JSON below from the MCP CLI tool and summarize results or errors cleanly.
User Question:
{original_question}
---
MCP JSON Response:
{raw_response}
"""
    llm = ChatBedrockConverse(
        model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",
        region_name="us-east-1",
        performance_config={"latency": "optimized"},
        max_tokens=4096,
        temperature=0.1,
        top_p=1,
        stop_sequences=["\n\nHuman"]
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()

# --- Utility ---
def clean_repo_name(display_name):
    return display_name.split(" (private)")[0]

# --- GitHub Data Fetching ---
def get_user_repos(username):
    if username == "Select user":
        return []
    query_dict = {"query": f"org:{username}", "perPage": 20, "page": 1}
    query_json = json.dumps(query_dict)
    try:
        result = subprocess.run(['./mcp1_clean.sh', 'call', 'search_repositories', query_json], capture_output=True, text=True, check=True)
        lines = result.stdout.splitlines()
        data_line = next((line for line in lines if line.startswith('data:')), None)
        outer_json = json.loads(data_line[5:].strip())
        result_data = outer_json.get('result', {})
        if result_data.get('isError', False):
            inner_text = result_data.get('content', [{}])[0].get('text', '')
            st.error(f"Error fetching repos: {inner_text}")
            return []
        inner_text = result_data.get('content', [{}])[0].get('text', '')
        inner_json = json.loads(inner_text)
        return [f"{i['name']} (private)" if i.get('private') else i['name'] for i in inner_json.get('items', []) if i.get('name')]
    except Exception as e:
        st.error(f"Repo fetch error: {str(e)}")
        return []

def get_repo_branches(username, repo):
    if not username or not repo:
        return []
    query_dict = {"owner": username, "repo": repo, "perPage": 100, "page": 1}
    query_json = json.dumps(query_dict)
    try:
        result = subprocess.run(['./mcp1_clean.sh', 'call', 'list_branches', query_json], capture_output=True, text=True, check=True)
        lines = result.stdout.splitlines()
        data_line = next((line for line in lines if line.startswith('data:')), None)
        outer_json = json.loads(data_line[5:].strip())
        result_data = outer_json.get('result', {})
        if result_data.get('isError', False):
            inner_text = result_data.get('content', [{}])[0].get('text', '')
            st.error(f"Error fetching branches: {inner_text}")
            return []
        inner_text = result_data.get('content', [{}])[0].get('text', '')
        inner_json = json.loads(inner_text)
        items = inner_json if isinstance(inner_json, list) else inner_json.get('items', [])
        return [i['name'] for i in items if 'name' in i]
    except Exception as e:
        st.error(f"Branch fetch error: {str(e)}")
        return []

# --- UI: User/Repo/Branch ---
st.markdown("## GitHub Repository Configuration")
colL, colR = st.columns([1, 1], gap="large")
with colL:
    user = st.selectbox("User Account *", USER_ACCOUNTS, index=USER_ACCOUNTS.index(st.session_state.user) if st.session_state.user in USER_ACCOUNTS else 0)
    st.session_state.user = user
with colR:
    repo_options = get_user_repos(user)
    repo_display = st.selectbox("Repo *", repo_options, index=repo_options.index(st.session_state.repo) if st.session_state.repo in repo_options else 0 if repo_options else None)
    repo = clean_repo_name(repo_display) if repo_display else None
    st.session_state.repo = repo
with colL:
    branches = get_repo_branches(user, repo) if repo else []
    branch_index = branches.index(st.session_state.branch) if st.session_state.branch in branches else branches.index("main") if "main" in branches else 0
    branch = st.selectbox("Branch", branches, index=branch_index, disabled=not branches, placeholder="Select branch")
    st.session_state.branch = branch

st.markdown("---")

# --- UI: Chat Interface ---
st.markdown("## Ask a question about this repository")
for role, msg in st.session_state.chat:
    with st.chat_message(role):
        st.write(msg)

question = st.chat_input("Type your question…")
if question:
    st.session_state.chat.append(("user", question))
    with st.chat_message("user"):
        st.write(question)

    user, repo, branch = st.session_state.user, st.session_state.repo, st.session_state.branch

    is_github = any(k in question.lower() for k in ["repo", "pull", "issue", "file", "branch", "commit", "sha", "workflow"])

    if user == "Select user" or not repo or not branch:
        st.warning("Please select a valid user, repository, and branch.")
        st.stop()

    if is_github:
        enhanced_question = f"In GitHub repository '{user}/{repo}' on branch '{branch}', {question}"
        cli_line = github_mcp_command(enhanced_question).strip().strip('`')
        st.session_state.cli = cli_line
        st.info(cli_line)
        result = subprocess.run(cli_line, shell=True, capture_output=True, text=True)
        if not result.stdout:
            st.error("No output returned.")
            st.stop()
        try:
            mcp_resp_check = json.loads(result.stdout.split('data:')[-1].strip())
            if mcp_resp_check.get('result', {}).get('isError'):
                explanation = explain_mcp_output(result.stdout, enhanced_question)
            else:
                explanation = explain_mcp_output(result.stdout, enhanced_question)
        except Exception as e:
            explanation = f"Parsing error: {e}"
        st.session_state.chat.append(("assistant", explanation))
        with st.chat_message("assistant"):
            st.write(explanation)
    else:
        casual = "Hi there! Let me know if you want to generate GitHub MCP commands."
        st.session_state.chat.append(("assistant", casual))
        with st.chat_message("assistant"):
            st.write(casual)

# --- Optional: Reset Button ---
if st.sidebar.button("Reset Session"):
    for k in ["chat", "cli", "last_question", "user", "repo", "branch"]:
        st.session_state[k] = [] if k == "chat" else ""
    st.rerun()
