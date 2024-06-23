import requests
from bs4 import BeautifulSoup
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Function to fetch the webpage content
def fetch_webpage_content(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }  # Adding a user-agent header to simulate a browser visit
    response = requests.get(url, headers=headers)
    return response.content

# Function to parse the HTML content
def parse_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    headers = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])  # Find all header tags
    return headers

# Function to extract headlines from parsed HTML (headers in this case)
def extract_headlines(headers):
    headlines = []
    for header in headers:
        headline = header.get_text().strip()
        if headline:  # Check if headline is not empty
            headlines.append(headline)
    return headlines

# Function to perform sentiment analysis
def analyze_sentiment(headlines):
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()
    return [sia.polarity_scores(headline)['compound'] for headline in headlines]

# Function to visualize sentiment distribution
def visualize_sentiment(sentiments):
    plt.hist(sentiments, bins=20, edgecolor='black')  # Adjust number of bins as needed
    plt.title('Sentiment Distribution of Section Headers on Wikipedia')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.grid(True)  # Adding grid lines for better readability
    plt.show()

def main():
    url = 'https://en.wikipedia.org/wiki/Wikipedia'  # Wikipedia page URL
    html_content = fetch_webpage_content(url)
    headers = parse_html(html_content)
    headlines = extract_headlines(headers)

    # Storing data in a DataFrame (optional, for better structure)
    df = pd.DataFrame(headlines, columns=['section_header'])

    sentiments = analyze_sentiment(headlines)
    df['sentiment'] = sentiments

    visualize_sentiment(sentiments)

if __name__ == "__main__":
    main()
