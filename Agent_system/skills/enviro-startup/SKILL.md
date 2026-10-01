---
name: enviro-startup
description: Standard Semantic Scholar MCP setup. Checks venvs/services.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  Dependencies: "MCP Daemon must be running."
---
# ⚙️ Enviro-Startup Protocol
This skill automates the validated, mandatory sequence required to set up a compliant working session for the Semantic Scholar MCP.
## 📚 Setup Procedure (Mandatory Execution Order)
This sequence must be followed rigorously:

1.  **Bootstrap Step (Path Discovery):** Programmatically extract the absolute path to the virtual environment from `./config/settings.yaml`.
2.  **Environment Activation:** Activate the virtual environment using the extracted path:
    `source <PATH EXTRACTED FROM SETTINGS.YAML>/.venv/bin/activate`
3.  **Load Credentials:** Source the project secrets:
    `source ./config/.env`
4.  **Verification Sequence:** Confirm that all necessary global environment variables are set:
    *   `SCHOLAR_API_KEY`: Must be visible and valid.
    *   `SCHOLAR_API_URL`: Must be visible and point to the bulk API endpoint.
5.  **Process Verification:** Confirm the mcp server is running and visible:
    *   Verify the existence of the `semanticscholar` process. If not found, **STOP** and report the failure. If found, report that the process is running.

### 🚨 Compliance Note
Any deviation from this strict, sequential setup procedure will result in an invalid or non-compliant work session and must be logged immediately in the `DEV_AUDIT_LOG.md`.