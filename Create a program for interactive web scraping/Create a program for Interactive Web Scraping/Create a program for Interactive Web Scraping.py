import requests
from bs4 import BeautifulSoup

def scrape_website(url, selector):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        elements = soup.select(selector)
        if not elements:
            print("No elements found for the given selector.")
            return
        print(f"\nFound {len(elements)} elements:\n")
        for idx, elem in enumerate(elements, 1):
            text = elem.get_text(strip=True)
            print(f"{idx}. {text}\n{'-'*40}")
    except Exception as e:
        print(f"Error: {e}")

def main():
    print("=== Interactive Web Scraper ===")
    while True:
        url = input("\nEnter website URL (or 'exit' to quit): ").strip()
        if url.lower() == 'exit':
            break
        selector = input("Enter CSS selector for data to scrape: ").strip()
        scrape_website(url, selector)

if __name__ == "__main__":
    main()