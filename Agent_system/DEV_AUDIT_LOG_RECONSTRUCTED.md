# 📜 Development Audit and Architectural Decisions Log

[Date of Reconstruction]: September 09, 2026

## ⚠️ Critical Infrastructure Note (Environment Setup):
- **Package Manager:** Must use `pip3` instead of `pip`.

## 💾 General Development Audit Requirements & Principles:
- **Compliance Constraint:** The agent must prioritize logging all architectural decisions, attempted fixes for environment failures (e.g., path issues), and roadblocks encountered into a dedicated, persistent Development Log file (DEV_LOG.md). This log proves the continuous oversight of human-level development effort to meet academic transparency standards.
- **Logging Protocol:** At the start of every task or session, a development log entry must be created or updated in DEV_LOG.md. This entry MUST explicitly state the current date and include a brief log entry confirming the process has started and outlining the scope of work for the session, ensuring proactive compliance logging.

## 🧭 Architecture Decisions (Arch Decisions):

### 1. Orchestrator Agent Implementation
- **Decision:** Proposed implementation of an 'Orchestrator Agent' using a specialized LLM (Gemma-4) to intelligently route user instructions to the most appropriate specialized agent, tool, or skill within the system. This agent will guide the workflow and is critical for complex agent behavior.
- **Refinement:** The Orchestrator Agent was refactored to decouple decision logic from model execution. Implemented a dedicated `LocalGemma4Provider` component in `orchestrator_agent.py` to abstract local model interaction, making the agent's routing mechanism independent of the underlying LLM framework (e.g., Hugging Face, vLLM). This modular approach enhances testability and resilience.
- **Next Steps:** Implement actual model loading and inference in `LocalGemma4Provider`.

## 👤 User Profile & Constraints (Source: User Profile Memory):
- **User Name:** Janet.
- **Role:** PhD student.
- **Timezone:** New Zealand.
- **Constraint 1 (Safety):** Must not perform any action (especially internet usage) without explicit user request and approval.
- **Constraint 2 (Ethics):** All work must adhere to ethical considerations.
- **Logging Requirement:** Requires explicit logging of all functional requirements, architectural decisions, and iterative development steps to be remembered and logged as compliance evidence in the final audit report.

## 🧩 System Context (Source: Codebase):
- **Project:** `semanticscholar-MCP-Server`
- **Purpose:** Implements a Model Context Protocol (MCP) server for interacting with the Semantic Scholar API, providing tools to search papers, retrieve details, and fetch citations.
- **Core Components:**
    - Client Agent (Calling Agent)
    - MCP Server (`semantic_scholar_server.py`) - Orchestrates the process.
    - Search Logic (`semantic_scholar_search.py`) - Performs API translation and calls.
    - External Service: Semantic Scholar API - Source of academic data.
- **Workflow:** Client -> MCP Server -> Search Logic -> Semantic Scholar API -> Search Logic -> MCP Server -> Client.

---
### 🗓️ [Current System Checkpoint]:
*Audit Log Update*: The system architecture was reviewed and a diagram was requested. The best practice for logging and recovery protocols was confirmed. The file is now marked as the authoritative source for development audit history.