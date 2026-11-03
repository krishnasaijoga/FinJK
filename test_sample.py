from dotenv import load_dotenv

from providers.news_provider import fetch_company_news

load_dotenv()

articles = fetch_company_news(
    ticker="AAPL",
    company_name="Apple",
    days=7,
)

for article in articles:
    print(article["title"])
    print(article["published_at"])
    print(article["source"])