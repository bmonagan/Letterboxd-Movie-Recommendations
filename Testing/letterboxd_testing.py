import requests
from bs4 import BeautifulSoup
from cosine_similarity import letter_boxd_get_recommendations

# Fetch the webpage

user_name = "breeban"
url = f"https://letterboxd.com/{user_name}/films/diary/"
response = requests.get(url)

# Parse the HTML content
soup = BeautifulSoup(response.text, "html.parser")

# Extract specific data (e.g., all links)
links = [a['href'] for a in soup.find_all('a', href=True)]
soup.prettify()  # Format the HTML for better readability

lb_watched = []
for a in soup.find_all('a', href=True):
    href = a['href']
    if href.startswith(f'/{user_name}/film/'):
        href = href.split('/')[3]  # Extract the film ID from the URL
        href = href.replace("-", " ")
        lb_watched.append(href)


print(lb_watched)

