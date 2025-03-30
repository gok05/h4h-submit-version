import wikipedia
import pandas as pd
import requests
from bs4 import BeautifulSoup

def extract_table_data(url):
    # Send a GET request to the Wikipedia page
    response = requests.get(url)
    response.raise_for_status()

    # Create a BeautifulSoup object to parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the table(s) on the page
    tables = soup.find_all('table')

    # Extract data from each table
    table_data = []
    for table in tables:
        # Find all table rows
        rows = table.find_all('tr')

        # Extract data from each row
        for row in rows:
            # Find all table cells (columns)
            cells = row.find_all(['th', 'td'])

            # Extract text from each cell
            cell_data = []
            for cell in cells:
                # Check if the cell contains a reference tag
                ref_tag = cell.find('sup', class_='reference')
                if ref_tag:
                    # Extract the reference number and remove it from the cell text
                    ref_num = ref_tag.get_text(strip=True)
                    cell_text = cell.get_text(strip=True)
                    cell_text = cell_text.replace(ref_num, '')

                    # Append the cell text and reference number to the cell data
                    cell_data.append(cell_text)
                    cell_data.append(ref_num)
                else:
                    cell_data.append(cell.get_text(strip=True))
                    cell_data.append('')  # Add an empty reference number if no reference is found

            # Append the cell data to the table data
            table_data.append(cell_data)

    return table_data

data = pd.read_excel(r"C:\Users\USER\Documents\Github\h4h-submit version\INPUT DATA FOR MODELS\for_scraping.xlsx")
titles = []
contents = []
table_data_list = []

total_names = len(data['ScientificName'])
for index, name in enumerate(data['ScientificName']):
    try:
        page = wikipedia.page(name)
        titles.append(page.title)
        contents.append(page.content)

        # Extract table data from the Wikipedia page
        table_data = extract_table_data(page.url)
        table_data_list.append(table_data)

    except wikipedia.exceptions.DisambiguationError as e:
        titles.append('Ambiguous')
        contents.append(str(e.options))
        table_data_list.append([])  # No table data available for ambiguous pages

    except wikipedia.exceptions.PageError:
        titles.append('Not Found')
        contents.append('')
        table_data_list.append([])  # No table data available for pages not found

    # Track progress
    progress = (index + 1) / total_names * 100
    print(f'Progress: {progress:.2f}%. Rows processed: {index + 1}/{total_names}')

scraped_data = pd.DataFrame({'ScientificName': data['ScientificName'], 'Title': titles, 'Content': contents, 'TableData': table_data_list})
scraped_data.to_excel(r"C:\Users\USER\Documents\Github\h4h-submit version\OUTPUT DATA OF MODELS\scraped_data.xlsx", index=False)






