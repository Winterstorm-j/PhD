---
name: development-audit-logging
description: Manages the safe, verifiable, and compliant update of critical, structured single-file documents (e.g., feature flags, development audit logs, project manifests). All modifications must preserve existing content structure and historical data while incorporating new information.
---
# Development Audit Logging

**Goal:** Manages the safe, verifiable, and compliant update process for structured single-file documents (e.g., development audit logs).

**Purpose:** To enforce a robust engineering process for documentation updates, recognizing that the execution environment's file system API is non-atomic for these tasks when using simple patching.

## Workflow (The 4-Phase Audit Cycle)

1.  **Pre-Check & Snapshot:**
    *   Use `read_file` to read the *entire* existing content of the target file, capturing a structural marker (e.g., line count, file size, or a unique section header) as the **BEFORE** snapshot.
2.  **Execute Change (The Core Action):**
    *   Use `patch` with extreme care. Always try to anchor the change using a unique, highly contextual `old_string` pattern (e.g., specific sentence fragments, unique IDs, or surrounding markdown headers) to minimize risk.
    *   If patching fails multiple times due to context mismatch, the **mandatory fallback** is to stop, read the file content, and require manual verification before any further tool execution can be attempted.
3.  **Post-Check & Validation (THE CRITICAL STEP):**
    *   After the patch attempt, use `read_file` again to capture the **AFTER** snapshot.
    *   **Validation Script:** A programmatic check must compare the structural markers (size, line count, presence of required headers) between the BEFORE and AFTER states. If they are identical, flag the success suspiciously and prompt for manual verification. *The success of the process is validated by the verifiable difference in the file's metadata.*
4.  **Finalization:**
    *   If validation passes, the change is logged, and the process is considered complete.

## Pitfalls & Anti-Patterns
*   **Never Assume Atomicity:** The file system API for large markdown updates is unreliable for complex, multi-step appends; multiple attempts must be treated as potential failure states that require human-level detective work.
*   **Path Management:** Use the `memory` tool's SSOT pattern to store the absolute file path for the log (e.g., `Agent_system/DEV_AUDIT_LOG.md`).
*   **Atomic Updates:** Treat the entire process as a single transaction. If any step fails, the entire operation must be rolled back or halted until the root cause (e.g., API limitation, transient lock) is understood.

## Support Files
*   `references/file_system_api_pitfalls.md`: A transcript detailing failed patch attempts and the resulting procedural block to prevent repeated, harmful attempts.
---