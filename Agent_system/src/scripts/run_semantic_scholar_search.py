import os
import requests
import time
import json
from datetime import datetime
from typing import Dict, List, Any

# --- Configuration and Constants ---
MAX_RETRIES = 5
INITIAL_BACKOFF_DELAY = 2 # seconds
INITIAL_QUERY = "DNA transfer" # Defining the constant globally
# Using the designated bulk API endpoint for compliance
SEARCH_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"

OUTPUT_FILE = "Agent_system/data/dna_transfer_records.json"
AUDIT_LOG_PATH = "Agent_system/DEV_AUDIT_LOG.md"

# --- Utility Functions ---
def get_env_variable(key: str) -> str:
    '''Safely reads a variable from the environment, or raises an error if vital.'''
    value = os.getenv(key)
    if not value:
        # This is a critical failure point. Raising a specific error is best practice.
        raise EnvironmentError(f"FATAL: Mandatory environment variable '{key}' not found in the environment.")
    return value

def retry_api_call(api_func, *args, max_retries=MAX_RETRIES, initial_delay=INITIAL_BACKOFF_DELAY):
    """
    Implements exponential backoff and retries for network calls experiencing
    rate limiting (HTTP 429) or temporary server failures (HTTP 5xx).
    """
    for attempt in range(max_retries):
        try:
            return api_func(*args)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                wait_time = initial_delay * (2 ** attempt)
                print(f"Rate limit hit (429). Retrying in {wait_time:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            elif e.response.status_code >= 400 and e.response.status_code < 500:
                print(f"API Error: Client error {e.response.status_code}. Aborting retry loop.")
                raise # Re-raise client errors immediately
            elif e.response.status_code >= 500:
                wait_time = initial_delay * (2 ** attempt)
                print(f"Server error {e.response.status_code}. Retrying in {wait_time:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise # Re-raise unknown errors
        except requests.exceptions.RequestException as e:
            print(f"Network/Connection Error: {e}. Retrying in {initial_delay * (2 ** attempt)} seconds...")
            time.sleep(initial_delay * (2 ** attempt))
        except Exception as e:
            print(f"An unexpected error occurred during retry: {e}")
            return None # Stop on unexpected error
            
    return None # Return None after all retries fail

def search_papers_bulk(api_key: str, api_url: str, query: str, limit: int) -> List[Dict[str, Any]]:
    """
    Searches for papers using the Semantic Scholar Bulk API, incorporating retry logic.
    Handles initial POST search and subsequent pagination GET requests.
    """
    headers = {"x-api-key": api_key, "Content-Type": "application/json"}
    
    def perform_search():
        # 1. Initial POST call to initiate the search and get the first page token
        payload = { 
            "query": query, "publicationTypes": "Review,JournalArticle,CaseReport,Conference,Dataset,Editorial,LettersAndComments,Study,Book,BookSection", 
            "fieldsOfStudy": "Computer Science,Medicine,Chemistry,Biology,Materials Science,Physics,Geology,Engineering,Environmental Science,Law",
            "fields": 
                "paperId,corpusId,externalIds,url,title,abstract,venue,publicationVenue,year,citationCount,influentialCitationCount,isOpenAccess,openAccessPdf,fieldsOfStudy,s2FieldsOfStudy,publicationTypes,publicationDate,journal,citationStyles,authors"} 
        response = requests.get(api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        search_data = response.json()
        
        if 'papers' not in search_data:
            print(f"API response lacks 'papers' key or failed: {search_data}")
            return [], None

        all_papers = []
        initial_papers = search_data['papers']
        
        # 2. Collect all results, handling pagination
        all_papers.extend(initial_papers)
        current_token = search_data.get('nextPageToken')
        
        while current_token:
            # Subsequent calls use GET with the token
            token_response = requests.get(f"{api_url}?token={current_token}", headers=headers, timeout=30)
            token_response.raise_for_status()
            token_data = token_response.json()

            if 'papers' in token_data:
                all_papers.extend(token_data['papers'])
                current_token = token_data.get('nextPageToken')
            else:
                break # Stop if token structure is unexpected

        # 3. Re-structure the retrieved data from the bulk response format
        structured_results = []
        for paper in all_papers:
            structured_results.append({
                paper
            })
        return structured_results

    # Execute the search with the retry wrapper
    return retry_api_call(perform_search)


def simulate_sem_search(query: str, output_file: str) -> List[Dict[str, Any]]:
    '''
    Attempts to process a bulk search for academic papers, implementing exponential backoff 
    and rate-limit recovery logic to retrieve maximum possible data by calling 
    the live Semantic Scholar API via requests.
    '''
    all_results = []
    # Get credentials and URLs from the environment
    try:
        SCHOLAR_API_KEY = get_env_variable("SCHOLAR_API_KEY")
        SCHOLAR_API_URL = get_env_variable("SCHOLAR_API_URL")
    except EnvironmentError as e:
        print(e)
        return []
    
    print("STEP 1/3: Preparing environment and logging audit trail.")
    try:
        # Using 'a' (append) mode to ensure audit log persists across subsequent runs
        with open(AUDIT_LOG_PATH, "a") as f: 
            f.write(f"\n### [Search Run - {datetime.now().strftime('%Y-%m-%d %H:%M')}] Query: {query}\\n")
            f.write("Attempting scalable bulk retrieval via Semantic Scholar Bulk API endpoint.\\n")
            f.write("Rate-limit handling (Exponential Backoff) is implemented for maximum yield.\\n")
        print(f"SUCCESS: Audit log updated at {AUDIT_LOG_PATH}")
    except Exception as e:
        print(f"Warning: Could not write to audit log: {e}")
    

    # 2. Core Logic (Live API Integration)
    print("\\nSTEP 2/3: Attempting live API calls for data retrieval.")
    
    # Call the new, robust search function
    scientific_papers = search_papers_bulk(
        api_key=SCHOLAR_API_KEY, 
        api_url=SCHOLAR_API_URL, 
        query=query, 
        limit=50
    )
    
    # Store results
    all_results = scientific_papers
    
    if all_results:
        print(f"SUCCESS: Retrieved {len(all_results)} search results.")
    else:
        print("WARNING: Zero search results retrieved. Check API keys, URLs, and rate limits.")


    # 3. Write results to the designated output file
    print("\\nSTEP 3/3: Verifying file write integrity.")
    try:
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        
        # Write the final list of dictionaries/records
        record_bytes = json.dumps(all_results, indent=2)
        with open(output_file, "w") as f:
            f.write(record_bytes)
        print(f"FINAL SUCCESS: Saved {len(all_results)} records to {os.path.basename(output_file)}.")
        return all_results
    except Exception as e:
        print(f"CRITICAL FAILURE: Could not write final output file: {e}")
        return []


def main():
    # Ensure directory for output exists
    os.makedirs(os.path.dirname(OUTPUT_FILE) or '.', exist_ok=True)
    
    # Run the advanced search now that the environment is ready.
    simulate_sem_search(INITIAL_QUERY, OUTPUT_FILE)

if __name__ == "__main__":
    main()