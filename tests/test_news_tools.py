from tools.news_tools import analyze_sentiment, search_recent_news


def test_search_news():
    ticker="APPL"
    news=search_recent_news(ticker)
    
    assert news["ticker"]==ticker
    assert news["articles_found"]==3
    assert news["news"] is not None


def test_analyze_sentiment():
    ticker="APPL"
    senitment=analyze_sentiment(ticker)

    assert senitment["ticker"]==ticker
    assert senitment["status"]=="success"
    assert senitment["sentiment_score"] is not None