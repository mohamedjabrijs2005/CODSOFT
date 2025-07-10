# Interactive Web Scraper

This is a simple interactive web scraping tool built with Python, `requests`, and `BeautifulSoup`.  
It allows you to fetch and display data from any website using a CSS selector.

## Features

- Enter any website URL and CSS selector interactively.
- Fetch and display matching elements in a user-friendly format.
- Handles errors gracefully.

## Requirements

- Python 3.x
- `requests` library
- `beautifulsoup4` library

Install dependencies with:

```
pip install requests beautifulsoup4
```

## Usage

Run the script:

```
python "Create a program for Interactive Web Scraping.txt"
```

Follow the prompts:

1. Enter the website URL (e.g., `https://quotes.toscrape.com/`)
2. Enter the CSS selector (e.g., `.quote`)

**Example selectors:**

- Bing search results: `.b_algo h2`
- Quotes to Scrape: `.quote`
- Wikipedia paragraphs: `p`

Type `exit` to quit.

## Example

```
=== Interactive Web Scraper ===

Enter website URL (or 'exit' to quit): https://quotes.toscrape.com/
Enter CSS selector for data to scrape: .quote

Found 10 elements:

1. “The world as we have created it is a process of our thinking. It cannot be changed without changing our thinking.” — Albert Einstein
----------------------------------------
...
```

---
**Author**

MOHAMED JABRI J S.