# Agent Development Plan: AI Research Assistant
**Project Goal:** Develop an autonomous agent pipeline that takes case circumstances, queries Semantic Scholar for relevant papers (on trace transfer/persistence), and builds a predictive Bayesian Network model, all while adhering to strict enterprise security and academic compliance standards.

## 🎯 Development Objectives & Constraints
*   **Core Functionality:** Circumstances $\rightarrow$ S2 API Query $\rightarrow$ Structured Data $\rightarrow$ Bayesian Model.
*   **Enterprise Security:** Mandatory integration of Containerization, EntraID OAuth flow, and explicit credential handling/logging.
*   **Compliance:** All decisions and outputs must adhere to the guidelines set out in 'Generative Artificial Intelligence in Doctoral Research Guidelines – University of Auckland'.

## 🗺️ Development Roadmap (To-Do List)

### P01: Define Agent Core & Architecture (Security & Compliance)
*   **Goal:** Establish the secure architectural foundation.
*   **Tasks:**
    1. Outline required modular Python classes for separation of concerns (e.g., `AuthManager`, `ScholarClient`, `BayesianModel`).
    2. Set up an environment variable template (`.env` or similar mechanism) for all necessary credentials and configuration values.
    3. Write a mandatory **Compliance Adherence Checklist** module/class to ensure every stage logs documentation evidence (e.g., citation tracking, model limitations).
    4. Implement stubs for EntraID OAuth flow handling to prove connectivity without storing live secrets.

### P02: Implement Knowledge Retrieval Module
*   **Goal:** Programmatically fetch scholarly data based on input query/circumstances.
*   **Source:** Semantic Scholar API (or alternative proxy).
*   **Tasks:**
    1. Build a robust client function to take keywords (e.g., 'trace transfer', 'persistence').
    2. Handle API rate limiting and caching mechanisms gracefully.
    3. Implement structured parsing of complex metadata (Title, Authors, Abstract, DOI, Year) ensuring that the output is ready for academic citation standards.

### P03: Build Bayesian Network Model
*   **Goal:** Transform qualitative research findings into a quantifiable probabilistic model.
*   **Input Data:** Structured data from Phase 2 (`P02_SCHOLAR`).
*   **Tasks:**
    1. Select and integrate a specialized Python library (e.g., `pgmpy` or `PyMC`).
    2. Develop the core modeling logic to define nodes (concepts/variables) and edges (dependencies) based on thematic coherence found in the retrieved papers.
    3. Implement inference techniques (e.g., calculating conditional probabilities) necessary for a predictive output.

### P04: Integration & Orchestration Module
*   **Goal:** Sequence all components into a single, reliable workflow.
*   **Process:** `run_agent(case_circumstances)` $\rightarrow$ P02 -> P03.
*   **Tasks:**
    1. Create the main orchestration function (`run_agent`) that manages state flow and calls P02 before passing clean data to P03.
    2. Implement comprehensive, layered error handling (try/except blocks) such that failure in one module gracefully defaults or logs a detailed report rather than crashing the entire pipeline.
    3. Format the final output into a **Compliance-Compliant Report** detailing findings, model assumptions, and security steps taken.

---
*This plan is stored as PLAN.md in the project directory.*