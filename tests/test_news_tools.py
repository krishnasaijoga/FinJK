from unittest.mock import patch

import pytest

from tools.news_tools import (
    analyze_sentiment,
    search_recent_news,
)

# --------------------------------------------------
# FIXTURES
# --------------------------------------------------

@pytest.fixture
def mock_articles():
    return [
        {
            "title": "Company reports record profits and strong growth",
            "description": "Revenue increased significantly.",
        },
        {
            "title": "Company announces successful product launch",
            "description": "Investors welcome the positive results.",
        },
    ]


@pytest.fixture
def mock_news_response(mock_articles):
    return mock_articles


# --------------------------------------------------
# SEARCH RECENT NEWS TESTS
# --------------------------------------------------

def test_search_news_success(mock_news_response):
    with patch(
        "tools.news_tools.fetch_company_news",
        return_value=mock_news_response,
    ) as mock_fetch:

        result = search_recent_news(
            ticker="AAPL",
            company_name="Apple",
            days=5,
        )

    mock_fetch.assert_called_once_with(
        ticker="AAPL",
        company_name="Apple",
        days=5,
    )

    assert result["ticker"] == "AAPL"
    assert result["articles_found"] == 2
    assert len(result["news"]) == 2
    assert result["data_status"] is True
    assert result["source"] == "GNews"


def test_search_news_ticker_normalization():
    with patch(
        "tools.news_tools.fetch_company_news",
        return_value=[],
    ) as mock_fetch:

        result = search_recent_news(
            ticker=" aapl ",
            company_name="Apple",
        )

    assert result["ticker"] == "AAPL"
    assert mock_fetch.call_args.kwargs["ticker"] == "AAPL"


def test_search_news_empty_response():
    with patch(
        "tools.news_tools.fetch_company_news",
        return_value=[],
    ):
        result = search_recent_news(
            ticker="AAPL",
            company_name="Apple",
        )

    assert result["articles_found"] == 0
    assert result["news"] == []
    assert result["data_status"] is False


def test_search_news_optional_company_name():
    with patch(
        "tools.news_tools.fetch_company_news",
        return_value=[],
    ) as mock_fetch:

        search_recent_news(ticker="AAPL")

    mock_fetch.assert_called_once_with(
        ticker="AAPL",
        company_name=None,
        days=5,
    )


def test_search_news_custom_days():
    with patch(
        "tools.news_tools.fetch_company_news",
        return_value=[],
    ) as mock_fetch:

        search_recent_news(
            ticker="NVDA",
            company_name="NVIDIA",
            days=10,
        )

    assert mock_fetch.call_args.kwargs["days"] == 10


def test_search_news_provider_failure():
    with patch(
        "tools.news_tools.fetch_company_news",
        side_effect=ConnectionError("GNews unavailable"),
    ), pytest.raises(
        ConnectionError,
        match="GNews unavailable",
    ):
        search_recent_news(
            ticker="AAPL",
            company_name="Apple",
        )


# --------------------------------------------------
# SENTIMENT ANALYSIS TESTS
# --------------------------------------------------

def test_analyze_sentiment_bullish():
    articles = [
        {
            "title": "Excellent profits and strong growth",
            "description": "Outstanding performance and success.",
        }
    ]

    result = analyze_sentiment(articles)

    assert result["sentiment_score"] > 0
    assert result["overall_sentiment"] in (
        "BULLISH",
        "STRONGLY_BULLISH",
    )
    assert result["data_status"] is True


def test_analyze_sentiment_bearish():
    articles = [
        {
            "title": "Terrible losses and severe financial crisis",
            "description": "Investors fear devastating results.",
        }
    ]

    result = analyze_sentiment(articles)

    assert result["sentiment_score"] < 0
    assert result["overall_sentiment"] in (
        "BEARISH",
        "STRONGLY_BEARISH",
    )


def test_analyze_sentiment_neutral():
    articles = [
        {
            "title": "Company publishes quarterly schedule",
            "description": "The meeting takes place on Tuesday.",
        }
    ]

    result = analyze_sentiment(articles)

    assert result["overall_sentiment"] == "NEUTRAL"
    assert -0.05 < result["sentiment_score"] < 0.05


def test_analyze_sentiment_empty_articles():
    result = analyze_sentiment([])

    assert result["overall_sentiment"] == "NEUTRAL"
    assert result["sentiment_score"] is None
    assert result["articles_analysed"] == 0
    assert result["article_sentiments"] == []
    assert result["data_status"] is False


def test_analyze_sentiment_missing_text():
    articles = [
        {
            "title": None,
            "description": None,
        },
        {
            "title": "",
            "description": "",
        },
    ]

    result = analyze_sentiment(articles)

    assert result["articles_analysed"] == 0
    assert result["sentiment_score"] is None
    assert result["article_sentiments"] == []
    assert result["data_status"] is False


def test_analyze_sentiment_skips_empty_articles():
    articles = [
        {
            "title": "Excellent growth and strong earnings",
            "description": "Investors are happy.",
        },
        {
            "title": None,
            "description": None,
        },
    ]

    result = analyze_sentiment(articles)

    assert result["articles_analysed"] == 1
    assert len(result["article_sentiments"]) == 1
    assert result["data_status"] is True


def test_analyze_sentiment_article_scores(mock_articles):
    result = analyze_sentiment(mock_articles)

    assert result["articles_analysed"] == 2
    assert len(result["article_sentiments"]) == 2

    for article in result["article_sentiments"]:
        assert "title" in article
        assert "score" in article
        assert -1 <= article["score"] <= 1


def test_analyze_sentiment_average_matches_individual_scores(
    mock_articles,
):
    result = analyze_sentiment(mock_articles)

    scores = [
        article["score"]
        for article in result["article_sentiments"]
    ]

    expected_average = sum(scores) / len(scores)

    # Small tolerance because individual scores
    # may be rounded before averaging.
    assert result["sentiment_score"] == pytest.approx(
        expected_average,
        abs=0.001,
    )


def test_analyze_sentiment_score_range(mock_articles):
    result = analyze_sentiment(mock_articles)

    assert -1 <= result["sentiment_score"] <= 1
