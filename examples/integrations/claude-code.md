# Using agentdream with Claude Code

The full guide is [docs/CLAUDE_CODE.md](../../docs/CLAUDE_CODE.md). The short version:

```bash
pip install "git+https://github.com/systemagents-ai/agentdream"

# MCP server: Claude can run dreams, show reports and (when you ask) accept them
claude mcp add agentdream -- agentdream serve-mcp

# Auto-dream after every 3 sessions (dry run; review with `agentdream list --jobs-dir .agentdream/jobs`)
agentdream install-claude-hook --every 3

# By hand
PROJ=~/.claude/projects/-home-me-myrepo
agentdream dream --memory CLAUDE.md --sessions "$PROJ/*.jsonl" --max-sessions 20
agentdream dream --memory CLAUDE.md --sessions "$PROJ" --backend claude-cli   # your Claude Code login as the model
agentdream schedule --memory CLAUDE.md --sessions "$PROJ/*.jsonl" --idle 30m --new-sessions 5
```
