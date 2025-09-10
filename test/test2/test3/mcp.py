import os
import json
import logging
import requests
from difflib import SequenceMatcher

log = logging.getLogger()
log.setLevel("INFO")

MCP_URL = os.environ.get("MCP_URL", "http://10.212.2.54:5432/mcp")
MCP_VERSION = os.environ.get("MCP_VERSION", "2025-06-18")
HTTP_TIMEOUT = float(os.environ.get("HTTP_TIMEOUT", "20"))


# --------------------------
# MCP session management
# --------------------------
def get_session_id():
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "MCP-Protocol-Version": MCP_VERSION,
    }
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": MCP_VERSION,
            "capabilities": {"tools": {}, "roots": {}},
            "clientInfo": {"name": "python-script", "version": "1.0"},
        },
    }
    resp = requests.post(MCP_URL, headers=headers, json=payload, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.headers.get("Mcp-Session-Id")


# --------------------------
# Send MCP tool command
# --------------------------
def send_tool_command(session_id, method_name, arguments=None):
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "MCP-Protocol-Version": MCP_VERSION,
        "Mcp-Session-Id": session_id,
    }

    payload = {
        "jsonrpc": "2.0",
        "id": 100,
        "method": "tools/call",
        "params": {"name": method_name, "arguments": arguments or {}},
    }

    resp = requests.post(MCP_URL, headers=headers, json=payload, stream=True, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()

    raw_chunks = []
    full_text = []

    for line in resp.iter_lines():
        if not line or not line.strip().startswith(b"data:"):
            continue
        try:
            json_part = json.loads(line[5:].decode("utf-8"))
            raw_chunks.append(json_part)

            if method_name == "get_file_contents" and json_part.get("kind") == "tool_result":
                content = (
                    json_part.get("delta", {}).get("content")
                    or json_part.get("result", {}).get("content", "")
                )
                if isinstance(content, str):
                    full_text.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if "text" in part:
                            full_text.append(part["text"])
        except Exception as e:
            log.warning(f"Error parsing MCP stream line: {e}")
            continue

    return {"raw": raw_chunks, "text": "".join(full_text).strip()}


# --------------------------
# Compare helper
# --------------------------
def is_relevant(jira_text, file_content, threshold=0.5):
    ratio = SequenceMatcher(None, jira_text, file_content).ratio()
    return ratio >= threshold, ratio


# --------------------------
# Main logic
# --------------------------
def main():
    owner = "Lokesh35-finance"
    repo = "mcp-git"
    branch = "main"
    jira_requirement = "Implement feature X and validate API integration"

    session_id = get_session_id()
    print(f"Session ID: {session_id}")

    # Step 1: List root folder files
    args = {"owner": owner, "repo": repo, "path": "/", "ref": branch}
    result = send_tool_command(session_id, "get_file_contents", args)

    entries = []
    for chunk in result.get("raw", []):
        res = chunk.get("result", {})
        for c in res.get("content", []):
            if c.get("type") == "text" and c.get("text"):
                try:
                    entries = json.loads(c["text"])
                except Exception as e:
                    log.error(f"JSON parse error: {e}")
                break
        if entries:
            break

    # Step 2: Loop through root files
    for entry in entries:
        if entry.get("type") == "file":
            file_path = entry["path"]
            print(f"\n📄 Checking file: {file_path}")

            content = send_tool_command(
                session_id, "get_file_contents", {"owner": owner, "repo": repo, "path": file_path, "ref": branch}
            ).get("text", "")

            relevant, score = is_relevant(jira_requirement, content)
            if relevant:
                print(f"✅ MATCH ({score:.2f}) -> {file_path}")
            else:
                print(f"❌ SKIP ({score:.2f}) -> {file_path}")


if __name__ == "__main__":
    main()
