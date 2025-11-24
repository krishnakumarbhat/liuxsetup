import os

def combine_code_files(directory_path, output_file='main.txt'):
    """
    Combines the content of all specified code files in a directory and its subdirectories
    into a single output file.

    Args:
        directory_path (str): The path to the directory to search.
        output_file (str): The name of the output file.
    """
    # Define the list of code file extensions to look for
    code_extensions = ['.py', '.c', '.cpp', '.h', '.hpp', '.js', '.java', '.cs', '.go', '.rs']
    
    # Open the output file in write mode
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Walk through the directory and its subdirectories
        for root, _, files in os.walk(directory_path):
            for file in files:
                # Check if the file has one of the specified code extensions
                if any(file.endswith(ext) for ext in code_extensions):
                    file_path = os.path.join(root, file)
                    print(f"Processing: {file_path}")
                    try:
                        # Write the file header to the output file
                        outfile.write(f"{file}\n")
                        # Write the file content to the output file
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                            outfile.write(infile.read())
                        # Add a separator
                        outfile.write("\n\n---\n\n")
                    except IOError as e:
                        print(f"Error reading {file_path}: {e}")

if __name__ == "__main__":
    # Get the directory path from the user
    target_directory = input("Enter the directory path to process: ")
    
    # Check if the provided path is a valid directory
    if os.path.isdir(target_directory):
        combine_code_files(target_directory)
        print(f"\nAll relevant code files have been combined into 'main.txt'.")
    else:
        print(f"Error: The provided path '{target_directory}' is not a valid directory.")