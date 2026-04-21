"""
CSV to BibTeX Converter Script
--------------------------------
This Python script converts bibliographic data from a CSV file into BibTeX format, making it suitable for use with citation managers like Zotero or JabRef. It focuses on the "Article" reference type but can be extended to support other types as needed.

Features:
- Flexible handling of CSV fields with default fallbacks for missing data.
- Automatic generation of BibTeX keys based on author names and publication years.
- Error handling for missing or incomplete rows.
- Interactive field mapping with sample data display.
- Option to save field mappings for future use.
- Automatic masking of uppercase letters in fields like title, booktitle, and journal to preserve formatting during BibTeX rendering.
- Restricts user input to valid BibTeX field names to ensure compatibility.

Usage:
To run the script, use the following Python command in your terminal:

```bash
python csv2bib.py <input_csv_path> <output_bib_path>
```

Replace `<input_csv_path>` with the path to your CSV file containing bibliographic data and `<output_bib_path>` with the desired output file path for the BibTeX file.
"""

import csv
import sys
import os
import json

VALID_BIBTEX_FIELDS = {
    "author": "Required",
    "title": "Required",
    "journal": "Required",
    "year": "Required",
    "volume": "Optional",
    "number": "Optional",
    "pages": "Optional",
    "month": "Optional",
    "note": "Optional",
    "doi": "Non-standard",
    "issn": "Non-standard",
    "zblnumber": "Non-standard",
    "eprint": "Non-standard",
    "url": "Non-standard",
    "annote": "Non-standard",
    "crossref": "Non-standard",
    "institution": "Non-standard",
    "publisher": "Non-standard",
    "type": "Non-standard",
    "abstract": "Non-standard"
}

def validate_csv(csv_file):
    """
    Validate the CSV file to ensure it is properly formatted.

    Parameters:
        csv_file (str): Path to the input CSV file.

    Returns:
        bool: True if the CSV is valid, False otherwise.
    """
    try:
        with open(csv_file, newline="", encoding="utf-8-sig") as csvfile:
            reader = csv.reader(csvfile, quotechar='"', delimiter=',')
            headers = next(reader, None)
            if not headers:
                print("Error: CSV file is empty or does not contain headers.")
                return False

            for row_number, row in enumerate(reader, start=2):  # Start from 2 to account for headers
                if len(row) != len(headers):
                    print(f"Error: Row {row_number} does not match the header length: {row}")
                    return False
        return True
    except Exception as e:
        print(f"Error while validating CSV: {e}")
        return False

def mask_capitals(value):
    """
    Wrap capital letters with curly brackets to preserve capitalization in BibTeX.

    Parameters:
        value (str): The string to process.

    Returns:
        str: The processed string with masked capital letters.
    """
    return ''.join(f"{{{char}}}" if char.isupper() else char for char in value)

# Function to generate a BibTeX key from the first author's last name and year
def generate_bibtex_key(authors, year):
    """
    Generate a BibTeX key from the first author's last name and the year.
    """
    last_names = authors.split(",")[0].split()
    last_name = last_names[-1] if last_names else "unknown"
    return f"{last_name.lower()}{year}"

# Function to prompt the user for field mapping and save the mapping if needed
def get_field_mapping(csv_file):
    """
    Prompt the user to map CSV fields to BibTeX fields, with an option to save mappings.

    Parameters:
        csv_file (str): Path to the input CSV file.

    Returns:
        dict: A mapping of CSV fields to BibTeX fields.
    """
    mapping_file = "field_mapping.json"

    # Check if a saved mapping exists
    if os.path.exists(mapping_file):
        use_saved = input("Saved field mapping found. Use it? (y/n): ").strip().lower()
        if use_saved == 'y':
            with open(mapping_file, 'r', encoding='utf-8') as file:
                return json.load(file)

    # Read CSV headers and display sample data
    with open(csv_file, newline="", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile, quotechar='"', delimiter=',')
        fields = reader.fieldnames
        if not fields:
            raise ValueError("CSV file does not contain headers.")

        print("Detected CSV fields:")
        print(", ".join(fields))

        # Collect sample rows without exhausting the reader
        sample_rows = []
        for _ in range(3):
            try:
                sample_rows.append(next(reader))
            except StopIteration:
                break

        print("\nSample data:")
        for row in sample_rows:
            print(row)

        # Prompt user for field mappings
        print("\nEnter the BibTeX field for each CSV field. Leave blank to skip mapping this field from the CSV.")
        mapping = {}
        for field in fields:
            non_blank_examples = [row[field] for row in sample_rows if field in row and row[field].strip()]
            print(f"\nField: {field}\nExample values: {non_blank_examples if non_blank_examples else '[No non-blank examples available]'}")
            bib_field = input(f"For the CSV field, '{field}' type a BibTeX field to use \n(valid options: {', '.join(VALID_BIBTEX_FIELDS.keys())}): ").strip()
            if bib_field and bib_field in VALID_BIBTEX_FIELDS:
                mapping[field] = bib_field
            elif bib_field:
                print(f"\nInvalid BibTeX field '{bib_field}'. Skipping this mapping.")

        # Option to save the mapping
        save_mapping = input("Save this mapping for future use? (y/n): ").strip().lower()
        if save_mapping == 'y':
            with open(mapping_file, 'w', encoding='utf-8') as file:
                json.dump(mapping, file, ensure_ascii=False, indent=4)

        return mapping

# Function to convert CSV to BibTeX using user-defined mappings
def convert_csv_to_bibtex_with_mapping(csv_file, bibtex_file, field_mapping):
    """
    Convert a CSV file to BibTeX format using a user-defined field mapping.

    Parameters:
        csv_file (str): Path to the input CSV file.
        bibtex_file (str): Path to the output BibTeX file.
        field_mapping (dict): Mapping of CSV fields to BibTeX fields.
    """
    with open(csv_file, newline="", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile, quotechar='"', delimiter=',')
        header_length = len(reader.fieldnames)

        with open(bibtex_file, "w", encoding="utf-8") as bibtexfile:
            for row_index, row in enumerate(reader, start=1):
                try:
                    # Check if the row is malformed
                    if len(row) != header_length or None in row or any(k is None for k in row.keys()):
                        print(f"Skipping malformed row {row_index}: {row}")
                        continue

                    bibtex_entry = "@article{\n"
                    # Map CSV fields to BibTeX fields
                    for csv_field, bib_field in field_mapping.items():
                        value = row.get(csv_field, "").strip()
                        print(f"  Mapping: CSV '{csv_field}' -> BibTeX '{bib_field}' | Value: '{value}'")  # Debugging
                        if value:
                            # Mask capitals for relevant fields
                            if bib_field in ["title", "booktitle", "journal"]:
                                value = mask_capitals(value)
                            bibtex_entry += f"  {bib_field} = {{{value}}},\n"
                    bibtex_entry += "}\n\n"
                    bibtexfile.write(bibtex_entry)
                except Exception as e:
                    print(f"Error processing row {row_index}: {row}\n{e}")

# Main script execution
if __name__ == "__main__":
    # Ensure correct number of arguments
    if len(sys.argv) != 3:
        print("Usage: python csv2bib.py <input_csv_path> <output_bib_path>")
    else:
        input_csv_path = sys.argv[1]
        output_bib_path = sys.argv[2]

        # Validate the CSV file
        if not validate_csv(input_csv_path):
            print("CSV validation failed. Please correct any errors in the CSV file and try again.")
            sys.exit(1)

        # Get field mapping and convert the file
        field_mapping = get_field_mapping(input_csv_path)
        convert_csv_to_bibtex_with_mapping(input_csv_path, output_bib_path, field_mapping)
