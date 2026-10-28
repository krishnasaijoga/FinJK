import logging
import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from strands import Agent
from strands.models.openai import OpenAIModel

from tools.financial_tools import compute_key_ratios, get_balance_sheet
from tools.news_tools import search_recent_news

load_dotenv()  # reads GROQ_API_KEY from a .env file, if present

logger=logging.getLogger(__name__)

MODEL_ID = os.getenv("GROQ_MODEL_ID", "openai/gpt-oss-120b")

class RiskAnalysisOutput(BaseModel):
    ticker:str
    overall_risk:Literal[
        "LOW",
        "MODERATE",
        "ELEVATED",
        "HIGH",
        "UNKNOWN"
    ]
    risk_score:float|None=Field(default=None,ge=0.0,le=1.0, description="Risk severity score, not loss probability.")
    financial_risks:list[str]
    business_risks:list[str]
    regulatory_risks:list[str]
    key_risk_metrics:dict[str,float|None]
    summary:str
    data_status:bool
    

SYSTEM_PROMPT = """
ROLE:
You are RiskAnalysisAgent, a financial investment risk
analyst specializing in company-specific risks.

OBJECTIVE:
Assess the potential risks associated with investing
in a company's stock using financial statements,
financial ratios, and recent company news.

INPUTS:
You will receive a company's stock ticker as a string.
Use the uppercase ticker consistently.

TOOLS:
1. get_balance_sheet:
   Retrieve assets, liabilities, shareholders' equity,
   cash balances, and debt.

2. compute_key_ratios:
   Retrieve profitability, leverage, and cash flow ratios.

3. search_recent_news:
   Retrieve recent company-related news articles.

EXECUTION RULES:
1. Call get_balance_sheet to retrieve financial position.
2. Call compute_key_ratios to obtain financial ratios.
3. Call search_recent_news to retrieve recent developments.
4. Evaluate financial, business, and regulatory risks.
5. Produce an evidence-based overall risk assessment.

Use all three tools when data is available.
Do not repeat tool calls unnecessarily.

RISK ASSESSMENT:
Classify risk using these qualitative guidelines:

    HIGH:
        Multiple significant financial or business risks,
        such as excessive leverage, severe liquidity concerns,
        or major regulatory threats.

    ELEVATED:
        Meaningful financial or business risks that could
        negatively affect company performance, even when
        some financial indicators remain healthy.

    MODERATE:
        Manageable financial and business risks with no clear
        evidence of severe financial distress.

    LOW:
        Relatively healthy financial conditions, manageable
        debt, and no major identified business or regulatory
        threats in the available data.

RISK CATEGORIES:
    financial_risks:
        Identify risks related to debt, liquidity,
        profitability, and cash generation.

    business_risks:
        Identify risks related to competition, customer
        demand, operations, and industry conditions.

    regulatory_risks:
        Identify risks related to government regulations,
        legal disputes, and compliance concerns.

RISK SCORE:
    risk_score represents estimated risk severity
    on a scale from 0.0 to 1.0.

    A higher score represents greater assessed risk.

    The score is NOT a probability of losing money.
    Do not fabricate numerical precision.
    If the available evidence cannot support a score,
    return null.

KEY RISK METRICS:
    Include relevant financial ratios and metrics
    that support the assessment.

    Use exact numerical values returned by the tools.
    Do not independently recalculate financial figures.

DATA AVAILABILITY:
    Set data_status to true only when sufficient
    financial and news data is available to support
    the requested assessment.

    Otherwise, set data_status to false and explain
    the limitations in the summary.

CONSTRAINTS:
- Use only the registered tools.
- Do not fabricate financial figures or news.
- Do not assume unavailable values are zero.
- Do not treat missing news as proof of low risk.
- Do not present mocked information as verified live data.
- Do not provide buy, sell, or hold recommendations.
- Do not confuse investment risk with stock valuation.
- Preserve financial metrics exactly as returned.
- Do not silently correct conflicting information.
- Do not predict future stock movements with certainty.
- Base conclusions only on evidence available from tools.

OUTPUT:
Return a structured RiskAnalysisOutput containing:

- ticker
- overall_risk
- risk_score
- financial_risks
- business_risks
- regulatory_risks
- key_risk_metrics
- summary
- data_status

The summary should briefly explain the overall
risk classification, the most important risk factors,
and any significant limitations in the available data.
"""


def create_risk_analysis_agent()->Agent:
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
    logger.info("Initializing RiskAnalysisAgent with model: %s", MODEL_ID)

    return Agent(
        name="RiskAnalysisAgent",
        model=model,
        # callback_handler=None,
        system_prompt=SYSTEM_PROMPT,
        tools=[search_recent_news, get_balance_sheet, compute_key_ratios],
        structured_output_model=RiskAnalysisOutput
    )


if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)

    agent=create_risk_analysis_agent()

    try:
        result=agent("Analyze risk for ticker APPL.")

        output=result.structured_output

        if output is None:
            raise ValueError("Agent returned no structured output")

        print(output.model_dump_json(indent=2))

    except Exception:
        logger.exception("Risk Analysis failed.")
        raise