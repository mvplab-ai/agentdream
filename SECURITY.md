# Security policy

## Supported versions

The `main` branch and the latest tagged release (currently `0.1.0`) are supported for security fixes.

## Reporting a vulnerability

Please **do not** open a public issue for a security report.

Use GitHub's private vulnerability reporting on this repository:

**[Report a vulnerability](https://github.com/mvplab-ai/agentdream/security/advisories/new)**

Include:

* a description of the issue and the impact
* steps to reproduce, or a proof of concept
* the agentdream version or commit, Python version, and OS
* whether any secret, transcript, or memory-store data could leave the machine or appear in a report

We will acknowledge the report and follow up through the advisory. Please give us a reasonable window to fix and release a patch before any public disclosure.

If private reporting is unavailable, open a [draft security advisory](https://github.com/mvplab-ai/agentdream/security/advisories) or contact the maintainers through the `mvplab-ai` GitHub organization.

## What we already guarantee

A dream never writes the input store. Secrets are redacted before analysis and again over every prompt and report quote. Model output is treated as untrusted. See [README.md](README.md#safety-guarantees) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#safety-guarantees).
