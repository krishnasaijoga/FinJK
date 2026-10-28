import logging
import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from strands import Agent
from strands.models.openai import OpenAIModel

from tools.news_tools import analyze_sentiment, search_recent_news

load_dotenv()  # reads GROQ_API_KEY from a .env file, if present

logger=logging.getLogger(__name__)

MODEL_ID = os.getenv("GROQ_MODEL_ID", "openai/gpt-oss-120b")

class NewsSentimentOutput(BaseModel):
    ticker:str
    overall_sentiment:Literal[
        "STRONGLY_BULLISH",
        "BULLISH",
        "NEUTRAL",
        "BEARISH",
        "STRONGLY_BEARISH"
    ]
    sentiment_score:float |None=Field(default=None,ge=-1.0,le=1.0)
    articles_analysed:int=Field(ge=0,description="Number of news articles actually analyzed")
    key_positive_factors:list[str]
    key_negative_factors:list[str]
    summary:str
    data_status:bool

SYSTEM_PROMPT="""
ROLE: You are a News sentiment analysis agent.

OBJECTIVE: Given a company ticker, you will use search_recent_news tool to search internet feed for recent(last one week) trending articles regarding the company. 
    Get sentiment from analyze_sentiment tool. 

INPUTS: Ticker for company in string format

TOOLS: search_recent_news, analyze_sentiment

CONTRAINTS:
- Do not use tools outside scope
- Do not invent data if no data is available
- When extracting numbers verify from source that they are exactly matching

OUTPUT: Returns a json containing ticker,
    overall_sentiment: <STRONGLY_BULLISH|BULLISH|NEUTRAL|BEARISH|STRONGLY_BEARISH>, 
    sentiment_score: score between -1 to +1,
    articles_analysed: number of articles analyzed, 
    key_positive_factors: key positive findings from articles, 
    key_negative_factors: key negative findings from articles, 
    summary: overall short summary from the findings, 
    data_status: True if data is avaialble else False
"""


def create_news_sentiment_agent()->Agent:
    model=OpenAIModel(
        client_args={
                "api_key": os.environ["GROQ_API_KEY"],
                "base_url": os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),  # or a local OpenAI-compatible server
                "max_retries": 5,  # retry dropped connections (the client default is 2)
            },
            model_id=MODEL_ID,
            # Groq reasoning models stream their thinking by default; hide it so only the answer prints.
            params={"extra_body": {"include_reasoning": False},
                    "max_tokens":4000
        }
    )

    print(f"Using {MODEL_ID} on Groq")
    logger.info("Initializing NewsAndSentimentAgent with model: %s", MODEL_ID)

    return Agent(
        name="NewsAndSentimentAgent",
        model=model,
        # callback_handler=None,
        system_prompt=SYSTEM_PROMPT,
        tools=[analyze_sentiment,search_recent_news],
        structured_output_model=NewsSentimentOutput
    )


if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)

    agent=create_news_sentiment_agent()

    try:
        result=agent("Analyze news sentiment for ticker APPL.")

        output=result.structured_output

        if output is None:
            raise ValueError("Agent returned no structured output")

        print(output.model_dump_json(indent=2))

    except Exception:
        logger.exception("News sentiment analysis failed.")
        raise