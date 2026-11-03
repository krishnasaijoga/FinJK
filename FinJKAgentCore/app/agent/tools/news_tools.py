from copy import deepcopy
from threading import RLock

from cachetools import TTLCache, cached

from strands import tool
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from providers.news_provider import fetch_company_news


_news_cache=TTLCache(maxsize=128,ttl=600)
_news_lock=RLock()


@cached(cache=_news_cache,lock=_news_lock)
def _search_recent_news_cached(ticker:str, company_name:str="", days:int=5)->dict:
    ticker = ticker.strip().upper()

    articles=fetch_company_news(
        ticker=ticker,
        company_name=company_name or None,
        days=days
    )

    return {
        "ticker":ticker.strip().upper(),
        "articles_found":len(articles),
        "news":articles,
        "data_status":bool(articles),
        "source":"GNews"
    }


@tool
def search_recent_news(ticker:str, company_name:str="", days:int=5)->dict:
    """Retrieve news articles about a company
    
    Args:
        ticker: ticker symbol for a company whose stock analysis query is searching. Eg:NVDA, APPL.
        company_name: Full company name for accurate search.
        days: Lookback period.
        
    Returns:
        Dictionary containing company news articles"""
    result=_search_recent_news_cached(
        ticker.strip().upper(),
        company_name.strip().lower(),
        days
    )
    return deepcopy(result)


@tool
def analyze_sentiment(articles:list[dict])->dict:
    """
    Return sentiment analysis for a company.

    Args:
        articles: Normalized articles with title and description fields.

    Returns:
        Dictionary containing sentiment results.
    """

    if not articles:
        return {
            "overall_sentiment":"NEUTRAL",
            "sentiment_score":None,
            "articles_analysed":0,
            "article_sentiments":[],
            "data_status":False
        }

    analyser=SentimentIntensityAnalyzer()

    scores=[]
    article_sentiments=[]

    for article in articles:
        text=" ".join(
            filter(
                None,
                [
                    article.get("title"),
                    article.get("description")
                ]
            )
        )

        if not text.strip():
            continue

        score=analyser.polarity_scores(text)["compound"]
        scores.append(score)

        article_sentiments.append({
            "title":article.get("title"),
            "score":round(score,4)
        })

    if not scores:
        return {
            "overall_sentiment": "NEUTRAL",
            "sentiment_score": None,
            "articles_analysed": 0,
            "article_sentiments":[],
            "data_status": False
        }

    average=sum(scores)/len(scores)

    if average>=0.5:
        sentiment="STRONGLY_BULLISH"
    elif average >= 0.05:
        sentiment = "BULLISH"
    elif average <= -0.5:
        sentiment = "STRONGLY_BEARISH"
    elif average <= -0.05:
        sentiment = "BEARISH"
    else:
        sentiment = "NEUTRAL"

    result={
        "overall_sentiment":sentiment,
        "sentiment_score":round(average,4),
        "articles_analysed":len(scores),
        "article_sentiments":article_sentiments,
        "data_status":True
    }

    return result