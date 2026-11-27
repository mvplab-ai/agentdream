# GitHub Action: dry-run dream

Copy-paste composite action that runs `agentdream dream` in CI, uploads `report.md` and `report.json`, and leaves the memory store unchanged unless you opt in.

The default backend is `deterministic`. That path needs no API key and no network call to a model.

## What it does

- Installs agentdream (the checkout that contains this action, when you use it from this repo).
- Runs a **dry run**. It does not call `agentdream accept` and it does not pass `--apply`.
- Writes job state to `jobs-dir` (default `$RUNNER_TEMP/agentdream-jobs`), outside the checkout.
- Stages the new store and the report under `<jobs-dir>/staged`, also outside the memory store.
- Uploads `report.md` and `report.json` as the `agentdream-report` artifact, including when a later check fails the step.

`true`, `yes`, and `1` are not enough to write the store. The only value of `apply` that runs `agentdream accept` is:

```text
I-understand-this-writes-the-memory-store
```

Anything else fails the step before a dream starts. The action still does not commit or push. Publishing the written store is a separate step you add yourself.

Set `fail-on-corrections: true` to fail the job when the report's corrected count is greater than zero. The dream stays a dry run, and the report is still uploaded. The files in `examples/memory` do contain corrections, so that flag fails on purpose against the bundled example.

## Permissions and jobs-dir

```yaml
permissions:
  contents: read
```

`contents: read` is enough. The job checks out the repo and does not push. Job files go to `jobs-dir`, which must be writable by the runner and must not be the memory store or a directory inside it (or the other way around). On GitHub-hosted runners the default `$RUNNER_TEMP/agentdream-jobs` satisfies that. Artifact upload uses the Actions runtime token; this repo's CI uploads artifacts with the same `contents: read` permission.

Opting into `apply` writes the memory files in the workspace. A later step that commits those files needs `contents: write`. This action does not add that step.

## Sample workflow

This is the workflow checked in at [`.github/workflows/dream-example.yml`](../../../.github/workflows/dream-example.yml). It runs on pull requests and on `workflow_dispatch`.

```yaml
name: Dream example

on:
  workflow_dispatch:
  pull_request:

permissions:
  contents: read

jobs:
  dry-run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: ./examples/integrations/github-action
        with:
          memory: examples/memory
          sessions: examples/sessions
          backend: deterministic
          jobs-dir: ${{ runner.temp }}/agentdream-jobs
```

From another repository, after this folder is on the default branch:

```yaml
permissions:
  contents: read

jobs:
  dry-run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: mvplab-ai/agentdream/examples/integrations/github-action@main
        with:
          memory: memory
          sessions: sessions
          backend: deterministic
          jobs-dir: ${{ runner.temp }}/agentdream-jobs
```

`sessions` is one file, one directory, or one glob. Put the glob in the input so the workflow shell does not expand it. A directory is read recursively.

```yaml
sessions: "logs/*.jsonl"
```

## Anthropic

`backend: anthropic` reads `ANTHROPIC_API_KEY` from the environment. Pass it from secrets. It is not an action input, so it is not stored as an input value. If the variable is missing, the action stops before calling the API.

```yaml
- uses: mvplab-ai/agentdream/examples/integrations/github-action@main
  with:
    memory: memory
    sessions: sessions
    backend: anthropic
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

Other backends (`openai`, `ollama`, `lmstudio`, `claude-cli`) are passed through to the CLI. The default remains `deterministic`.

## Inputs

| Input | Default | Purpose |
|---|---|---|
| `memory` | required | Store path |
| `sessions` | required | One transcript file, directory, or glob |
| `backend` | `deterministic` | `deterministic`, `anthropic`, `openai`, `ollama`, `lmstudio`, `claude-cli` |
| `model` | empty | Model name for an LLM backend |
| `jobs-dir` | `$RUNNER_TEMP/agentdream-jobs` | Job state. Must not overlap the memory store |
| `out` | `<jobs-dir>/staged` | Staged store and report |
| `python-version` | `3.12` | Interpreter used to install and run |
| `install` | `auto` | `auto`, `local`, `git`, or `skip` |
| `fail-on-corrections` | `false` | Fail when `corrected` > 0. Still a dry run |
| `apply` | empty | Only `I-understand-this-writes-the-memory-store` writes the store |
| `extra-args` | empty | Extra `agentdream dream` arguments. `--apply` is rejected |
| `artifact-name` | `agentdream-report` | Artifact name. Change it if you call the action twice in one job |
| `upload-artifact` | `true` | Upload `report.md` and `report.json` |

## Outputs

`job-id`, `status`, `corrected`, `report-md`, `report-json`, `jobs-dir`.

```yaml
- id: dream
  uses: ./examples/integrations/github-action
  with:
    memory: examples/memory
    sessions: examples/sessions
    jobs-dir: ${{ runner.temp }}/agentdream-jobs
- run: echo "${{ steps.dream.outputs.job-id }} corrected=${{ steps.dream.outputs.corrected }}"
```
