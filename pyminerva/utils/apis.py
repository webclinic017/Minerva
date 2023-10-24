import feedparser
import pandas as pd
import datetime as dt
from datetime import datetime

# List of RSS feed URLs
rss_feed_urls = [
    "https://www.oecd-ilibrary.org/rss/content/collection/mei-data-en/latest?fmt=rss",
]

print('--------------')
print(datetime.now())

def fetch_rss_data(url):
    feed = feedparser.parse(url)
    print("Feed Title:", feed.feed.title)
    for entry in feed.entries:
        if entry.title == 'Consumer prices':
            start_time = dt.datetime(2020, 1, 1)
            end_time = dt.datetime(2022, 2, 1)
            print("Entry Title:", entry.title)
            print("Entry Link:", entry.link)
            print("Entry Published Date:", entry.published)
            print("Entry Summary:", entry.summary)
            print("\n")
            df = web.DataReader('PRICES_CPI', 'oecd', start_time, end_time)
    else:
        continue


# Fetch data from multiple RSS feeds
for url in rss_feed_urls:
    fetch_rss_data(url)