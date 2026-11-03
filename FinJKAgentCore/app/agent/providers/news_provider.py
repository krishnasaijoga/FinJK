import os
from datetime import UTC, datetime, timedelta

import requests

GNEWS_URL="https://gnews.io/api/v4/search"

def fetch_company_news(
        ticker:str,
        company_name:str|None=None,
        days:int=3,
        max_articles:int=5
)->list[dict]:
    """Retreive recent company news from GNews"""

    if days<=0:
        raise ValueError("days must be positive")

    if not 1<=max_articles<=5:
        raise ValueError("max_articles must be 1-5")

    api_key=os.getenv("GNEWS_API_KEY")

    if not api_key:
        raise ValueError("GNEWS_API_KEY is not configured")

    ticker=ticker.strip().upper()

    if not ticker:
        raise ValueError("Ticker cannot be empty")

    query=f'"{company_name}"' if company_name else f'"{ticker}"'

    start=datetime.now(UTC) - timedelta(days=days)

    response=requests.get(
        GNEWS_URL,
        params={
            "q":query,
            "lang":"en",
            "max":max_articles,
            "from":start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "apikey":api_key
        },
        timeout=15
    )

    response.raise_for_status()

    articles=response.json().get("articles",[])

    result_articles= [
        {
            "ticker":ticker,
            "title":article.get("title"),
            "description":article.get("description"),
            "published_at":article.get("publishedAt"),
            "source":article.get("source",{}).get("name"),
            "url":article.get("url")
        } for article in articles if article.get("title")
    ]

    return result_articles