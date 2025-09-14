# MCP GitHub Agent

An intelligent GitHub assistant powered by MCP (Model Context Protocol) that can explore repositories, analyze files, and generate refined follow-up questions automatically.

## 🚀 Features
- **MCP Tool Invocations**  
  - `get_file_contents` → Fetch exact contents of files.  
  - `list_files_recursive` → Traverse entire repo structures recursively.  
  - `list_branches` → Explore branch structures.  
  - `search_repositories` → Find repositories using advanced search queries.  

- **Next Question Generation**  
  Automatically proposes a context-aware follow-up question after each MCP command to guide further analysis.  

- **Dependency Extraction**  
  Detects imports, functions, and cross-file references.  
  Summarizes file logic in concise sentences.  

- **Content Normalization**  
  Handles user variations like *"fetch/print/show/open/display file"* and always returns the raw file content when requested.  

## 📂 Example Usage
### 1. Fetch file contents
**User:**  
