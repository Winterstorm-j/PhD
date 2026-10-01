import os
import requests
import time
import json
from argparse import ArgumentParser
from typing import Dict, List, Any
from datetime import datetime
from urllib.parse import urlparse, urlunparse, urlencode

# --- Configuration and Constants ---
MAX_RETRIES = 5
INITIAL_BACKOFF_DELAY = 2 # seconds
INITIAL_QUERY = "DNA transfer" # Defining the constant globally
# Using the designated bulk API endpoint for compliance
SEARCH_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"

OUTPUT_FILE = "Agent_system/data/dna_transfer_records.json"
AUDIT_LOG_PATH = "/Users/jbuc045/Projects/PhD/Agent_system/DEV_AUDIT_LOG.md"

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

def search_papers_bulk(api_key: str, api_url: str, query: str) -> List[Dict[str, Any]]:
    """
    Searches for papers using the Semantic Scholar Bulk API, incorporating retry logic.
    Handles initial GET search and subsequent pagination GET requests.
    """
    headers = {"x-api-key": api_key, "Content-Type": "application/json"}
    
    def perform_search():
        """
        Searches for papers using the Semantic Scholar Bulk API, incorporating retry logic.
        Handles initial GET search and subsequent pagination GET requests.
        """
        print("--- Initializing Search URL Construction ---")
        
        # Define the required parameters that should be passed via the URL as GET parameters
        # These are based on a successful Postman GET call.
        initial_params = {"query": query, 
            "publicationTypes": "Review,JournalArticle,CaseReport,Conference,Dataset,Editorial,LettersAndComments,Study,Book,BookSection", 
            "fieldsOfStudy": "Computer Science,Medicine,Chemistry,Biology,Materials Science,Physics,Geology,Engineering,Environmental Science,Law",
            "fields": "paperId,corpusId,externalIds,url,title,abstract,venue,publicationVenue,year,citationCount,influentialCitationCount,isOpenAccess,openAccessPdf,fieldsOfStudy,s2FieldsOfStudy,publicationTypes,publicationDate,journal,citationStyles,authors"
        }

        # Build the initial URL with query parameters to avoid whitespace issues
        url_parts = list(urlparse(api_url.strip()))
        url_parts[4] = urlencode(initial_params)

        # Rebuild it perfectly
        initial_url = urlunparse(url_parts)

        headers = {"x-api-key": api_key, "Content-Type": "application/json", 
            "Host": "api.semanticscholar.org",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive"}
    
        # 1. Initial GET call to initiate the search and get the first page token
        print(f"Attempting initial search GET to: {initial_url[:100]}...")
        response = requests.get(initial_url, headers=headers, timeout=30)
        response.raise_for_status()
        search_data = response.json()
    
        all_papers = []
        initial_papers = search_data.get('data')
        initial_count = len(initial_papers)
        print(f"[API Call 1] Initial search retrieved {initial_count} records.")
        all_papers.extend(initial_papers)
    
        current_token = search_data.get('token')
    
        # 2. Collect all results, handling pagination
        while current_token:
            # Subsequent calls use GET with the token
            print(f"Fetching next page using token: {current_token[:50]}...")
            url_parts[5] = urlencode({"token": current_token})
            
            new_url = urlunparse(url_parts)
            time.sleep(1.1)  # Guaranteed delay to enforce a minimum 1.1s gap between page requests
            token_response = requests.get(new_url, headers=headers, timeout=30)
            token_response.raise_for_status()
            token_data = token_response.json()
        
            new_records = token_data.get('data')
            new_count = len(new_records)
            print(f"[API Call N] Subsequent page retrieved {new_count} records.")

            if new_records:
                all_papers.extend(new_records)
                current_token = token_data.get('token')
                
                if len(all_papers) > 10000:
                    break
            else:
                break # Stop if token structure is unexpected

        # 3. Re-structure the retrieved data from the bulk response format
        structured_results = [
            paper for paper in all_papers
        ]
        return structured_results

    # Execute the search with the retry wrapper
    return retry_api_call(perform_search)


def simulate_sem_search(query: str, output_file: str, api_key: str, api_url: str) -> List[Dict[str, Any]]:
    '''
    Attempts to process a bulk search for academic papers, implementing exponential backoff 
    and rate-limit recovery logic to retrieve maximum possible data by calling 
    the live Semantic Scholar API via requests.
    '''
    all_results = []
    
    print("STEP 1/3: Preparing environment and logging audit trail.")
    try:
        length_of_query = len(all_results) if all_results else 0
        
        # Using 'a' (append) mode to ensure audit log persists across subsequent runs
        with open(AUDIT_LOG_PATH, "a") as f: 
            f.write(f"\n### [Search Run - {datetime.now().strftime('%Y-%m-%d %H:%M')}] Query: {query}\\n")
            f.write(f"{length_of_query} records retrieved.\\n")
        print(f"SUCCESS: Audit log updated at {AUDIT_LOG_PATH}")
    except Exception as e:
        print(f"Warning: Could not write to audit log: {e}")
    

    # 2. Core Logic (Live API Integration)
    print("\\nSTEP 2/3: Attempting live API calls for data retrieval.")
    
    # Call the new, robust search function
    scientific_papers = search_papers_bulk(
        api_key=api_key, 
        api_url=api_url, 
        query=query
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
        #  CRITICAL MANUAL DEBUGGING POINT START 
        print(f"CRITICAL FAILURE: File write I/O error occurred: {e}")
        # Force an exception to halt the process and expose the true nature of the failure.
        raise IOError(f"Failed to write output file for manual investigation: {e}") # Re-raise the error
        #  CRITICAL MANUAL DEBUGGING POINT END 

def main():
    parser = ArgumentParser(description="Searches for academic papers using the Semantic Scholar Bulk API.")
    parser.add_argument("--query", type=str, required=True, help="The search query (e.g., 'DNA transfer').")
    parser.add_argument("--api-key", type=str, required=True, help="The Semantic Scholar API Key (x-api-key header).")
    parser.add_argument("--api-url", type=str, required=True, help="The Semantic Scholar Bulk API URL.")
    parser.add_argument("--output-file", type=str, default="data/search_results.json", help="Path to save the JSON output.")
    
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    
    # Run the advanced search now that the environment is ready.
    simulate_sem_search(args.query, args.output_file, args.api_key, args.api_url)

if __name__ == "__main__":
    main()