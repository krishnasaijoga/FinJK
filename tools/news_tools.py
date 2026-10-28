from strands import tool


@tool
def search_recent_news(ticker:str)->dict:
    """Retrieve news articles about a company
    
    Args:
        ticker: ticker symbol for a company whose stock analysis query is searching. Eg:NVDA, APPL.
        
    Returns:
        Dictionary containing company news articles"""

    mock_news={
        "APPL":[
            {
                "id":1,
                "headline":"APPL shows strong Q3 revenue growth",
                "source":"News Today",
                "date":"2026-10-26",
                "summary":"APPL reports strong demand for AI chips."
            },
            {
                "id":2,
                "headline":"APPL faces export restrictions",
                "source":"Market News",
                "date":"2026-10-21",
                "summary":"New export restrictions could impact international sales."
            },
            {
                "id": 3,
                "headline": "APPL announces developer conference",
                "source": "Technology News",
                "date": "2026-10-22",
                "summary": "APPL announces its upcoming developer event."
            }
        ]
    }

    articles=mock_news.get(ticker.upper(),[])

    return {
        "ticker":ticker,
        "articles_found":len(articles),
        "news":articles
    }


@tool
def analyze_sentiment(ticker:str)->dict:
    """
    Return sentiment analysis for a company.

    Args:
        ticker: ticker symbol for a company whose stock analysis query is searching. Eg:NVDA, APPL.

    Returns:
        Dictionary containing sentiment results.
    """
    mock_sentiment={
        "APPL":{
            "overall_sentiment":"Neutral",
            "sentiment_score":0.05,
            "positive_articles":1,
            "negative_articels":1,
            "neutral_articles":1,
            "summary":"Strong AI demand is offset by potential export restrictions."
        }
    }

    result=mock_sentiment.get(ticker.upper())
    if result is None:
        return {
            "ticker":ticker.upper(),
            "status":"no_mock_data",
            "overall_sentiment":None,
            "sentiment_score":None,
            "summary":"No mock sentiment data avaialble."
        }

    return {
        "ticker":ticker.upper(),
        "status":"success",
        **result    # for unpacking dict
    }