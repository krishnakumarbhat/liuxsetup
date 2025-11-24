from bs4 import BeautifulSoup

# Define the file path
file_path = 'bookmarks.html'

try:
    # Open and read the HTML file
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    # Create a Beautiful Soup object
    soup = BeautifulSoup(html_content, 'lxml')

    # Now you can work with the parsed HTML. 
    # For example, let's find all the links (<a> tags).
    links = soup.find_all('a')
    
    # Print the links found
    if links:
        print("Found the following links:")
        for link in links:
            print(f"Text: {link.get_text()} | URL: {link.get('href')}")
    else:
        print("No links found in the file.")

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please check the file path.")
except Exception as e:
    print(f"An error occurred: {e}")