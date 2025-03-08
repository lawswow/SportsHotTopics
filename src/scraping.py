import requests
#from bs4 import BeautifulSoup
import os
import time

def fetch_sports_articles(url_list, out_dir="data/raw"):

    #Fetch articles from a list of URLs and save their raw HTML to local storage.
    #url_list: List of article URLs to scrape
    #out_dir: Directory for storing raw HTML files

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        # if it doesn’t exist, creates the folder path

    for i, url in enumerate(url_list): #i == index
        try: #"try" to handle possible network errors gracefully
            response = requests.get(url) #fetches the webpage by sending an HTTP GET request to the URL.
            response.raise_for_status() #If server returns an error code, this will raise an exception.
            html_content = response.text #contains the raw HTML as a string
            filename = f"article_{i}.html"
            filepath = os.path.join(out_dir, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html_content)

            print(f"Saved {url} to {filepath}")
            time.sleep(1)  #avoid hitting rate limits
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
            #displays error type if failure

if __name__ == "__main__":
    # urls
    urls = [
        "https://www.bbc.co.uk/sport",
        "https://www.bbc.com/sport/football",
        "https://www.bbc.co.uk/sport/rugby-union",
        "https://www.bbc.co.uk/sport/cricket",
        "https://www.bbc.co.uk/sport/tennis",
        "https://www.bbc.co.uk/sport/golf",
        "https://www.skysports.com",
        "https://www.skysports.com/football",
        "https://www.skysports.com/rugby-union",
        "https://www.skysports.com/cricket",
        "https://www.skysports.com/tennis",
        "https://www.skysports.com/golf",
        "https://www.theguardian.com/uk/sport",
        "https://www.theguardian.com/football",
        "https://www.theguardian.com/sport/rugby-union",
        "https://www.theguardian.com/sport/cricket",
        "https://www.theguardian.com/sport/tennis",
        "https://www.theguardian.com/sport/golf",
        "https://www.mirror.co.uk/sport/",
        "https://www.mirror.co.uk/sport/football/",
        "https://www.mirror.co.uk/sport/rugby-union/",
        "https://www.mirror.co.uk/sport/cricket/",
        "https://www.mirror.co.uk/sport/golf/"
    ]
    fetch_sports_articles(urls)
