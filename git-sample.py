# This script creates a ready-to-use "github-repo-scripts" folder with handy Git/GitHub utilities,
# then zips it so the user can download and copy into any repo.

import os, textwrap, json, zipfile, pathlib, sys, stat

base = "/mnt/data/github-repo-scripts"
scripts_dir = os.path.join(base, "scripts")
os.makedirs(scripts_dir, exist_ok=True)

def write_executable(path, content):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content.lstrip())
    # make executable
    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IEXEC)

def write_file(path, content):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content.lstrip())

# README
readme = """
# GitHub Repo Utility Scripts

Drop this `scripts/` folder into any Git repository to speed up common maintenance tasks.

## What's inside

- **git-clean-merged-branches.sh** — Delete local/remote branches already merged into the default branch.
- **git-find-large-files.sh** — Find large Git blobs (e.g., trim repo bloat).
- **git-sync-fork.sh** — Rebase your fork from the upstream default branch.
- **release-bump.sh** — Bump semantic version in `VERSION`, tag, and push.
- **changelog-from-conventional.sh** — Generate a simple Markdown changelog from conventional commits since the last tag.
- **gh-protect-branch.py** — Enable branch protection using the GitHub API.
- **gh-pr-create.sh** — Create a PR via the `gh` CLI or GitHub REST API.
- **repo-audit.py** — Quick repo audit: default branch, contributors, file types, etc.

## Quick start

```bash
# From inside your git repo:
cp -r ./github-repo-scripts/scripts ./
chmod +x scripts/*.sh
