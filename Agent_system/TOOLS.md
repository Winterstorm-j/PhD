# TOOLS.md - Local Notes (Forensics Context)

Skills define _how_ tools work. This file is for _your_ specifics — the stuff that's unique to your setup: camera names and locations, SSH hosts and aliases, preferred TTS voices, speaker/room names, device nicknames, anything environment-specific to the lab or evaluation process.

## 🔬 Forensic Protocols & Equipment
**General Guideline:** All analysis must be repeatable. Assumptions regarding trace behavior must cite a reference DOI found in an approved literature search.
*   **Instrument A:** Standard serial number format is XXXXXXXXXX. Requires calibration log YYYYMMDD. Location: [Lab Room 3]. (Must check system logs for last cal date).
*   **Analysis Software:** Primary platform is XYZ v4.1. Must verify patch level against CISA advisories *before* running casework.

## 🔗 API & Authentication Protocols
This section details the external connections required for the Forensic Knowledge Graphing Service. *Credentials themselves must be stored in a secure vault, not here.*

### Semantic Scholar API Connection (Required)
- **Base Endpoint:** `https://api.semanticscholar.org/graph/...`
- **Authentication:** Requires $\text{API Key}$ and potentially access tokens managed by the Orchestration Layer.
- **Rate Limiting:** Must track calls to prevent 429 errors. Assume a baseline of 100 requests/minute until licensed otherwise.

### Authorization Gateway (Future State)
- **Service:** EntraID integration required for user authorization checks before accessing core services.
- **Mechanism:** Needs an explicit API wrapper that handles OAuth flow validation.

## 💾 Data & Process Structures
*   **Case Intake Format:** Standardized JSON structure: `{"crime_type": "...", "trace_type": "...", "activities": ["..."]}`.
*   **Output Structure Goal:** For every returned paper, the justification must address $\text{Transfer}$, $\text{Persistence}$, $\text{Prevalence}$, AND $\text{Recovery}$.

## 🌐 General System Notes
- **System Timezone/Time Reference:** Always default to UTC for forensic logging unless otherwise specified by law.