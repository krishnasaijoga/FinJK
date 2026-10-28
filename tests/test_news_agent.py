from unittest.mock import MagicMock, patch

import pytest

from agents.news_sentiment_agent import NewsSentimentOutput, create_news_sentiment_agent


@pytest.fixture
def mock_agent_response():
    """Generate a fake structured agent output."""
    return NewsSentimentOutput(
        ticker="APPL",
        overall_sentiment="BULLISH",
        sentiment_score=0.75,
        articles_analysed=3,
        key_positive_factors=[
            "Strong revenue growth",
            "Increasing demand for AI products",
        ],
        key_negative_factors=[
            "Regulatory uncertainty",
        ],
        summary=(
            "Positive financial performance is partially "
            "offset by regulatory concerns."
        ),
        data_status=True
    )


@patch("agents.news_sentiment_agent.Agent")
@patch("agents.news_sentiment_agent.OpenAIModel")
def test_create_news_sentiment_agent(
    mock_model,
    mock_agent,
    monkeypatch
):
    monkeypatch.setenv("GROQ_API_KEY","fake-test-key")

    agent=create_news_sentiment_agent()

    mock_model.assert_called_once()
    mock_agent.assert_called_once()

    assert agent==mock_agent.return_value


@patch("agents.news_sentiment_agent.Agent")
@patch("agents.news_sentiment_agent.OpenAIModel")
def test_news_sentiment_agent_response(
    mock_model,
    mock_agent,
    mock_agent_response,
    monkeypatch
):
    monkeypatch.setenv("GROQ_API_KEY","fake-test-key")
    fake_result=MagicMock()
    fake_result.structured_output=mock_agent_response

    fake_agent=mock_agent.return_value
    fake_agent.return_value=fake_result

    agent=create_news_sentiment_agent()

    result=agent("Analyze news sentiment for ticker APPL.")

    output=result.structured_output

    assert isinstance(output, NewsSentimentOutput)
    assert output.ticker=="APPL"
    assert output.overall_sentiment=="BULLISH"
    assert output.sentiment_score==0.75
    assert output.articles_analysed==3

    assert len(output.key_positive_factors)==2
    assert len(output.key_negative_factors)==1

    fake_agent.assert_called_once_with("Analyze news sentiment for ticker APPL.")


def test_invalid_sentiment_score():
    with pytest.raises(ValueError):
        NewsSentimentOutput(
            ticker="APPL",
            overall_sentiment="BULLISH",
            sentiment_score=1.5,
            articles_analysed=3,
            key_positive_factors=[],
            key_negative_factors=[],
            summary="Test Summary",
            data_status=True
        )