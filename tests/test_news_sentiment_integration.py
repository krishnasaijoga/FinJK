
import copy
import os
from unittest.mock import patch

import pytest
from dotenv import load_dotenv
from strands.hooks import BeforeToolCallEvent

from agents.news_sentiment_agent import (
    create_news_sentiment_agent,
    NewsSentimentOutput,
)

load_dotenv()


@pytest.fixture
def sample_articles():
    return [
        {
            "ticker": "AAPL",
            "title": "Apple reports strong revenue growth",
            "description": "The company announced record profits.",
            "published_at": "2026-10-28T10:00:00Z",
            "source": "Test News",
            "url": "https://example.com/article1",
        },
        {
            "ticker": "AAPL",
            "title": "Apple faces supply chain challenges",
            "description": "Rising costs may pressure profit margins.",
            "published_at": "2026-10-29T11:00:00Z",
            "source": "Test News",
            "url": "https://example.com/article2",
        },
    ]


@pytest.mark.integration
def test_news_agent_passes_articles_to_vader(sample_articles):
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not configured")

    captured_search_calls = []
    captured_sentiment_calls = []

    agent = create_news_sentiment_agent()

    # Capture real agent tool-call arguments.
    def capture_tool_call(event: BeforeToolCallEvent):
        tool_use = event.tool_use

        if tool_use["name"] == "search_recent_news":
            captured_search_calls.append(
                copy.deepcopy(tool_use["input"])
            )

        elif tool_use["name"] == "analyze_sentiment":
            captured_sentiment_calls.append(
                copy.deepcopy(tool_use["input"])
            )

    agent.add_hook(
        capture_tool_call,
        BeforeToolCallEvent,
    )

    # Patch the provider used by search_recent_news,
    # without replacing the registered Strands tools.
    with patch(
        "tools.news_tools.fetch_company_news",
        return_value=sample_articles,
    ) as mock_provider:
        result = agent(
            "Analyze recent news sentiment for ticker AAPL. "
            "First use search_recent_news, then pass the "
            "returned news articles to analyze_sentiment. "
            "Use only retrieved evidence and return "
            "NewsSentimentOutput."
        )

    # Check news retrieval.
    assert captured_search_calls, (
        "NewsAgent did not call search_recent_news"
    )
    assert mock_provider.called, (
        "News retrieval provider was not invoked"
    )

    # Check sentiment analysis.
    assert captured_sentiment_calls, (
        "NewsAgent did not call analyze_sentiment"
    )

    import json

    print("\nEXPECTED ARTICLES:")
    print(json.dumps(sample_articles, indent=2))

    print("\nACTUAL SENTIMENT TOOL INPUT:")
    print(json.dumps(captured_sentiment_calls, indent=2))

    # Verify exact article transfer.
    expected_articles = [
        {
            "title": article["title"],
            "description": article["description"],
        }
        for article in sample_articles
    ]

    assert any(
        call.get("articles") == expected_articles
        for call in captured_sentiment_calls
    ), (
        "Article titles and descriptions passed to VADER "
        "do not match the GNews results"
    )

    # Verify structured output.
    assert result.structured_output is not None

    output = NewsSentimentOutput.model_validate(
        result.structured_output
    )

    assert output.ticker == "AAPL"
    assert output.data_status is True

    print("\nNews calls:", captured_search_calls)
    print("\nSentiment calls:", captured_sentiment_calls)
    print("\nNews output:", output.model_dump_json(indent=2))
