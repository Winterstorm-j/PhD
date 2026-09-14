import graphviz
import sys
import os

def generate_dot_diagram(dot_source: str, output_filename: str, format: str = 'svg') -> str:
    """
    Generates and renders an architectural diagram from a DOT graph string.

    Args:
        dot_source: The string content in DOT format.
        output_filename: The name to save the base file name (without extension).
        format: The desired output image format (e.g., 'svg', 'png', 'pdf').

    Returns:
        The absolute path to the generated diagram file, or an error message.
    """
    try:
        # Use graphviz's Source object to handle the DOT graph
        dot = graphviz.Source(dot_source)
        
        # The .render() method saves the file and returns the path.
        # We need the directory to exist first.
        output_dir = os.path.dirname(os.path.abspath(output_filename))
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Renders the diagram and returns the path to the image file
        diagram_path = dot.render(
            filename=output_filename, 
            format=format, 
            view=False, # Do not automatically open the diagram
            cleanup=True # Clean up the temporary dot.gv file
        )
        return diagram_path

    except graphviz.backend.ExecutableNotFound(executable, directory):
        return f"Error: Graphviz executable not found. Please ensure 'graphviz' is installed and in your PATH. Run: 'sudo apt-get install graphviz' or similar for your OS."
    except Exception as e:
        return f"An unexpected error occurred during diagram generation: {e}"

def main():
    """Main entry point for the diagram generation script."""
    if len(sys.argv) < 3:
        print("Usage: python generate_diagram.py <input_dot_file> <output_base_name> [--format <svg|png|pdf>]")
        print("Example: python generate_diagram.py graph.dot my_system --format svg")
        sys.exit(1)

    input_file = sys.argv[1]
    output_base_name = sys.argv[2]
    
    format_arg = "--format"
    output_format = "svg"

    # Simple argument parsing to handle --format
    if len(sys.argv) > 3 and sys.argv[1] == "--format":
        try:
            # Look for the format after the flag
            output_format = sys.argv[2].lower()
        except IndexError:
            print("Error: --format requires a format type (svg|png|pdf).")
            sys.exit(1)


    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        sys.exit(1)

    try:
        with open(input_file, 'r') as f:
            dot_source = f.read()
    except Exception as e:
        print(f"Error reading input file {input_file}: {e}")
        sys.exit(1)

    # Generate and print the result/error message
    result_path = generate_dot_diagram(dot_source, output_base_name, output_format)
    
    if "Error:" in result_path:
        print(result_path)
    else:
        print(f"✅ Success! Diagram generated and saved to {result_path}")

if __name__ == "__main__":
    main()
