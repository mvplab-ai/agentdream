# Using agentdream with Claude Code

## As an MCP server

```bash
pip install agentdream
claude mcp add agentdream -- agentdream serve-mcp
# or project-scoped (writes .mcp.json):
claude mcp add --scope project agentdream -- agentdream serve-mcp
```

Then ask Claude: *"Dream over my memory in ~/.claude/projects/<proj>/memory using the last sessions in
~/.claude/projects/<proj>/ and show me the report."* It calls `dream`, `show_report`, and, once you
approve, `accept`.

## From the shell

```bash
PROJ=~/.claude/projects/-home-me-myrepo
agentdream dream --memory "$PROJ/memory" --sessions "$PROJ/*.jsonl" --max-sessions 20
agentdream show <job-id>
agentdream accept <job-id>            # backs up the original first
```

Claude Code session files are auto-detected: text blocks are kept, tool calls are
reduced to one line, failed tool results are kept (they reveal recurring mistakes),
thinking blocks and successful tool output are dropped.

To use your Claude Code login as the model instead of the heuristics:

```bash
agentdream dream --memory ./memory --sessions "$PROJ/*.jsonl" --backend claude-cli
```

## Scheduled dreaming

```bash
agentdream schedule --memory "$PROJ/memory" --sessions "$PROJ/*.jsonl" --idle 30m --new-sessions 5
```
