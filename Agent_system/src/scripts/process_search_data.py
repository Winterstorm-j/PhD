import json
import os

def process_data(input_file_path, output_true_path, output_false_path, search_term="DNA transfer"):
    """
    Reads a large JSON file containing academic search results, processes each record,
    and splits them into two separate JSON files: one for records containing
    the search term, and one for all others.
    
    Args:
        input_file_path (str): Path to the raw search results JSON file.
        output_true_path (str): File path to save records containing the search term.
        output_false_path (str): File path to save records that do not contain the search term.
        search_term (str): The specific phrase to filter by.
    """
    print(f"Starting data split process using search term: '{search_term}'")
    
    if not os.path.exists(input_file_path):
        print(f"ERROR: Input file not found at {input_file_path}. Exiting.")
        return

    all_results = []
    
    try:
        with open(input_file_path, 'r') as f:
            # The input file is expected to be a list of dictionaries/records.
            # Since the file was written by the main script using a list structure,
            # we attempt to load it as a list and iterate.
            try:
                all_results = json.load(f)
            except json.JSONDecodeError:
                print("ERROR: Failed to decode JSON file. Ensure the input is a valid JSON list.")
                return
    except Exception as e:
        print(f"ERROR: An unexpected file read error occurred: {e}")
        return

    records_with_term = []
    records_without_term = []
    
    print(f"Total records loaded: {len(all_results)}")
    
    for record in all_results:
        # Convert the record to string for search. This is robust for different field types.
        record_str = json.dumps(record)
        
        if search_term.lower() in record_str.lower():
            records_with_term.append(record)
        else:
            records_without_term.append(record)

    try:
        # Write the filtered lists back to JSON files
        with open(output_true_path, 'w') as f:
            json.dump(records_with_term, f, indent=2)
        print(f"✅ Successfully saved {len(records_with_term)} records containing '{search_term}' to {output_true_path}")

        with open(output_false_path, 'w') as f:
            json.dump(records_without_term, f, indent=2)
        print(f"✅ Successfully saved {len(records_without_term)} records NOT containing '{search_term}' to {output_false_path}")
    
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to write output files. Check permissions or disk space. Error: {e}")

if __name__ == "__main__":
    # Define standard paths relative to the project root
    INPUT_FILE = "data/searchresults021026.json"
    OUTPUT_TRUE_FILE = "data/dna_transfer_results.json"
    OUTPUT_FALSE_FILE = "data/other_results.json"
    SEARCH_TERM = "DNA recovery"

    process_data(INPUT_FILE, OUTPUT_TRUE_FILE, OUTPUT_FALSE_FILE, SEARCH_TERM)
