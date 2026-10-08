---
name: semantic-scholar-full-sync
description: Executes the full Semantic Scholar data retrieval and filtering workflow.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  Dependencies: MCP Daemon must be running.
  Critical Path: Execution must ensure the daemon is running before running the final script.
---
Usage:
# Workflow Steps:
1. **Startup Compliance:** Always runs `enviro-startup` logic (MCP Daemon and venv activation check).
2. **Data Acquisition:** Executes the API-intensive search on Semantic Scholar via the dedicated path in the .venv interpreter.
3. **Audit Logging:** Enforces logging to the absolute path: `/Users/jbuc045/Projects/PhD/Agent_system/DEV_AUDIT_LOG.md`.
4. **Post-Processing:** Executes a dedicated Python routine (e.g., `process_search_data.py`) to filter the raw data into specified categories, completing the analysis phase.

# Key Context / Pitfalls:
- Rate limiting: Implement delays (e.g., `time.sleep(1.1)`).
- Error Handling: Must contain robust multi-stage exception handling.
- Dependencies: Requires packages installed via the project's virtual environment.

# Use Case Example:
When processing a new query (e.g., "Neurodegenerative disease"), the core steps are:
1. Execute: `/Users/jbuc045/Projects/PhD/TPPRDB_Analysis/.venv/bin/python src/scripts/run_semantic_scholar_search.py --query "Neurodegenerative disease" [etc...]`
2. Execute: `python src/scripts/process_search_data.py data/searchresults_new.json`
