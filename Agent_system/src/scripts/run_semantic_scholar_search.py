import json
import os
import time
import requests # Added dependency for HTTP calls
from datetime import datetime
from typing import Dict, List, Any

# --- Configuration and Constants ---
MAX_RETRIES = 5
INITIAL_QUERY = "DNA transfer"
OUTPUT_FILE = "Agent_system/data/dna_transfer_records.json"
AUDIT_LOG_PATH = "Agent_system/DEV_AUDIT_LOG.md"

# --- Utility Functions ---
def get_env_variable(key: str) -> str:
    '''Safely reads a variable from the environment, or raises an error if vital.'''
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(f"Mandatory environment variable '{key}' not found in the environment.")
    return value

def simulate_sem_search(query: str, output_file: str) -> List[Dict[str, Any]]:
    '''
    Attempts to process a bulk search for academic papers, implementing exponential backoff 
    and rate-limit recovery logic to retrieve maximum possible data by calling 
    the live Semantic Scholar API via requests.
    '''
    all_results = []
    
    try:
        # 1. Load Credentials from Environment
        SCHOLAR_API_KEY = get_env_variable("SCHOLAR_API_KEY")
        SCHOLAR_API_URL = get_env_variable("SCHOLAR_API_URL")
    except EnvironmentError as e:
        print(f"FATAL SETUP ERROR: {e}")
        return []

    print("STEP 1/3: Preparing environment and logging audit trail.")
    try:
        with open(AUDIT_LOG_PATH, "w") as f:
            f.write(f"\\n### [Search Run - {datetime.now().strftime('%Y-%m-%d %H:%M')}] Query: {query}\\n")
            f.write("Attempting scalable bulk retrieval via Semantic Scholar Bulk API endpoint.\\n")
            f.write("Rate-limit handling (Exponential Backoff) is implemented for maximum yield.\\n")
        print(f"SUCCESS: Audit log created at {AUDIT_LOG_PATH}")
    except Exception as e:
        print(f"Warning: Could not write to audit log: {e}")

    # 2. Core Logic (Live API Integration)
    print("\\nSTEP 2/3: Attempting live API calls for data retrieval.")
    
    for attempt in range(MAX_RETRIES):
        try:
            print(f"--- Attempt {attempt + 1}/{MAX_RETRIES} ---")
            
            headers = {"x-api-key": SCHOLAR_API_KEY, "Content-Type": "application/json"}
            payload = {"query": query, "limit": 50} # Payload structure assumed for a real API call
            
            # 1. Construct the correct search URI based on documentation
            search_url = f"{SCHOLAR_API_URL}"
            # 2. Use a GET request and pass parameters via 'params' argument
            response = requests.get(search_url, headers=headers, params=payload, timeout=30)
            response.raise_for_status() # Raises HTTPError for 4xx/5xx status codes
   
            # Assuming the actual response data is a list of records
            with open(f"{OUTPUT_FILE}", "a") as file:
                while True:
                    if "data" in response:
                        retrieved += len(response["data"])
                        print(f"Retrieved {retrieved} papers...")
                        for paper in response["data"]:
                            print(json.dumps(paper), file=file)
                    # checks for continuation token to get next batch of results
                    if "token" not in response:
                        break
                    response = requests.get(f"{SCHOLAR_API_URL}&token={response['token']}").json()
                
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            print(f"API ERROR: HTTP {status_code} - {e.response.reason}.");
            if status_code == 429:
                wait_time = 2 ** attempt
                if attempt < MAX_RETRIES - 1:
                    print(f"Rate Limit Hit (429). Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise ConnectionError("Maximum retries reached for rate limit failure.")
            else:
                raise ConnectionError(f"API Client/Server Error: {status_code} ({e.response.reason})")
        except requests.exceptions.RequestException as e:
            print(f"Network/Connection Error: {e}. Retrying...")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                raise ConnectionError("Maximum retries reached for connection failure.")
        except json.JSONDecodeError:
            print("API ERROR: Failed to decode JSON from response. Check API documentation or endpoint.")
            break
        except Exception as e:
            print(f"An unexpected general error occurred during API interaction: {type(e).__name__} - {e}")
            break

    # 3. Write results to the designated output file
    print("\\nSTEP 3/3: Verifying file write integrity.")
    try:
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        final_records = all_results
        
        record_bytes = json.dumps(final_records, indent=2)
        with open(output_file, "w") as f:
            f.write(record_bytes)
        print(f"FINAL SUCCESS: Saved {len(final_records)} records to {os.path.basename(output_file)}.")
        return final_records
    except Exception as e:
        print(f"CRITICAL FAILURE: Could not write final output file: {e}")
        return []

if __name__ == "__main__":
    # Ensure directory for output exists
    os.makedirs(os.path.dirname(OUTPUT_FILE) or '.', exist_ok=True)
    
    # Run the advanced search
    simulate_sem_search(INITIAL_QUERY, OUTPUT_FILE)