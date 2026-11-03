import json
import logging
import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel
from strands import Agent
from strands.models.openai import OpenAIModel

from agents.financial_data_agent import FinancialDataOutput
from agents.news_sentiment_agent import NewsSentimentOutput
from agents.risk_agent import RiskAnalysisOutput
from agents.valuation_agent import (
    DCFValuation,
    PeerValuation,
    ValuationAnalysisOutput,
    ValuationMultiples,
)

load_dotenv()  # reads GROQ_API_KEY from a .env file, if present

logger=logging.getLogger(__name__)

MODEL_ID = os.getenv("GROQ_MODEL_ID", "openai/gpt-oss-120b")

class ReportWriterOutput(BaseModel):
    ticker: str
    investment_recommendation: Literal[
        "STRONG BUY",
        "BUY",
        "HOLD",
        "SELL",
        "STRONG SELL",
        "UNKNOWN"
    ]
    key_reasons:list[str]
    supporting_evidence:dict[str,str]
    summary: str
    data_status: bool


SYSTEM_PROMPT="""

ROLE:
You are a Report Writing agent that writes a summarized report based on the other agents output.

OBJECTIVE:
Assess the different summaries from other agents, recommend stock buy/sell, and write a one page summary note on why this recommendation makes sense.

INPUTS:
You will receive a company stock ticker.
Use the uppercase ticker consistently.

If financial data from FinancialDataAgent is provided,
use it as supporting context.

Do not assume financial data has been provided when
only a ticker is available.

If news sentiment data from NewsAndSentimentAgent is provided,
use it as supporting context.

Do not assume news sentiment data has been provided when
only a ticker is available.

If risk analysis data from RiskAgent is provided,
use it as supporting context.

Do not assume risk analysis data has been provided when
only a ticker is available.

If valuation analysis data from ValuationAgent is provided,
use it as supporting context.

Do not assume valuation analysis data has been provided when
only a ticker is available.

EXECUTION RULES:

1. Read the outputs from all four specialist agents.
2. Verify that all outputs refer to the requested ticker.
3. Review financial health, sentiment, risk, and valuation.
4. Identify agreements and contradictions between findings.
5. Use numerical metrics to support important conclusions.
6. Generate a recommendation based on the combined evidence.
7. Explain the key reasons supporting the recommendation.
8. Preserve the original specialist conclusions.

INVESTMENT RECOMMENDATION CLASSIFICATION:

STRONG BUY:
Strong financial fundamentals, significant valuation
upside, favorable sentiment, and manageable risks.
Multiple independent indicators strongly support
a favorable investment outlook.

BUY:
Healthy financial fundamentals and attractive valuation.
Some risks or negative indicators may exist, but the
overall evidence supports a favorable outlook.

HOLD:
Mixed financial, sentiment, risk, or valuation signals.
The stock may be fairly valued, or positive factors
may be balanced by significant concerns.

SELL:
Weak fundamentals, unfavorable valuation, or significant
business and financial risks.
The combined evidence suggests an unfavorable outlook.

STRONG SELL:
Multiple severe negative indicators, such as financial
distress, substantial overvaluation, and high risks.
The overall evidence strongly supports an unfavorable
investment outlook.

UNKNOWN:
Insufficient, missing, contradictory, or unreliable
data prevents a defensible investment recommendation.

DECISION RULES:
- Consider financial health, valuation, risk, and
  news sentiment together.
- Do not base the recommendation on one metric alone.
- Give appropriate importance to material risks.
- Explain conflicting signals between agents.
- An undervalued stock does not automatically mean BUY.
- Strong financial health does not automatically mean BUY.
- Negative news sentiment does not automatically mean SELL.
- If sufficient evidence is unavailable, return UNKNOWN.

These classifications are qualitative.
Do not invent fixed numerical thresholds.

DATA AVAILABILITY:

Set data_status to true only when sufficient
financial, news sentiment, risk, and valuation
information is available to support the final
investment recommendation.

Set data_status to false when essential information
is missing, inconsistent, or insufficient.

When data_status is false due to insufficient evidence,
set investment_recommendation to UNKNOWN.

Explain missing information in the summary.

CONSTRAINTS:

- Use only the outputs provided by specialist agents.
- Do not invent financial metrics, news, or valuations.
- Do not independently recalculate financial ratios or DCF.
- Do not alter numerical values received from other agents.
- Do not ignore conflicting findings.
- Do not assume missing data indicates favorable conditions.
- Do not present mocked data as verified investment research.
- If evidence is insufficient, return UNKNOWN.
- Do not confuse stock valuation with company financial health.
- Do not claim future stock performance is guaranteed.
- Treat recommendations as general research assessments,
  not personalized investment advice.

OUTPUT:

Return a structured ReportWriterOutput containing:

ticker:
The company ticker being analyzed.

investment_recommendation:
One of STRONG BUY, BUY, HOLD, SELL,
STRONG SELL, or UNKNOWN.

key_reasons:
Concise reasons supporting the final recommendation.

supporting_evidence:
A dictionary mapping each specialist analysis
to the important evidence supporting the conclusion.

summary:
A coherent investment research summary combining
financial health, sentiment, risk, and valuation.

data_status:
True only when sufficient reliable input data
is available to support the recommendation.
Otherwise False.

"""

def create_report_writer_agent()->Agent:
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
    logger.info("Initializing ReportWriterAgent with model: %s", MODEL_ID)
    
    return Agent(
        name="ReportWriterAgent",
        model=model,
        # callback_handler=None,
        system_prompt=SYSTEM_PROMPT,
        tools=[],
        structured_output_model=ReportWriterOutput
    )


def generate_investment_report(
    financial_data: FinancialDataOutput,
    news_data: NewsSentimentOutput,
    risk_data: RiskAnalysisOutput,
    valuation_data: ValuationAnalysisOutput,
) -> ReportWriterOutput:
    """Generate a final report from four specialist outputs."""

    # Validate that all analyses refer to the same company
    ticker = financial_data.ticker.upper()

    for output in [news_data, risk_data, valuation_data]:
        if output.ticker.upper() != ticker:
            raise ValueError(
                "Ticker mismatch between specialist agent outputs"
            )

    # Package specialist outputs
    report_input = {
        "ticker": ticker,
        "financial_analysis": financial_data.model_dump(
            mode="json"
        ),
        "news_sentiment_analysis": news_data.model_dump(
            mode="json"
        ),
        "risk_analysis": risk_data.model_dump(
            mode="json"
        ),
        "valuation_analysis": valuation_data.model_dump(
            mode="json"
        ),
    }

    # Create the ReportWriterAgent
    agent = create_report_writer_agent()

    # Invoke the agent using structured specialist data
    result = agent(
        "Generate a final fundamental analysis report "
        "using only the following specialist outputs:\n"
        + json.dumps(report_input, indent=2)
    )

    output = result.structured_output

    if output is None:
        raise ValueError(
            "ReportWriterAgent returned no structured output"
        )

    if output.ticker.upper() != ticker: # type: ignore
        raise ValueError(
            "ReportWriterAgent returned an incorrect ticker"
        )

    return output # type: ignore


# 1. FinancialDataAgent mock output
financial_output = FinancialDataOutput(
    ticker="APPL",
    overall_company_health="STRONG",
    period="annual",
    revenue=40000000,
    cost_of_revenue=22000000,
    operating_income=12000000,
    net_income=9500000,
    eps=6.25,
    total_assets=35000000,
    total_liabilities=28000000,
    shareholders_equity=7000000,
    cash_and_equivalents=5000000,
    total_debt=1000000,
    operating_cash_flow=11000000,
    investing_cash_flow=-2000000,
    financing_cash_flow=-7000000,
    capital_expenditure=1500000,
    free_cash_flow=9500000,
    net_profit_margin=0.2375,
    return_on_equity=1.3571,
    debt_to_equity=0.1429,
    free_cash_flow_margin=0.2375,
    data_status=True,
    summary=(
        "The company demonstrates healthy profitability, "
        "manageable debt, and strong cash generation."
    ),
)


# 2. NewsAndSentimentAgent mock output
news_output = NewsSentimentOutput(
    ticker="APPL",
    overall_sentiment="NEUTRAL",
    sentiment_score=0.05,
    articles_analysed=3,
    key_positive_factors=[
        "Strong revenue growth",
        "Increasing demand for new products",
    ],
    key_negative_factors=[
        "Potential regulatory restrictions",
        "Increasing competition",
    ],
    summary=(
        "Recent news indicates mixed sentiment. "
        "Positive financial developments are offset "
        "by regulatory and competitive concerns."
    ),
    data_status=True,
)


# 3. RiskAnalysisAgent mock output
risk_output = RiskAnalysisOutput(
    ticker="APPL",
    overall_risk="ELEVATED",
    risk_score=0.65,
    financial_risks=[
        "Potential pressure on future profit margins",
    ],
    business_risks=[
        "Increasing market competition",
    ],
    regulatory_risks=[
        "Potential regulatory restrictions",
    ],
    key_risk_metrics={
        "debt_to_equity": 0.1429,
        "net_profit_margin": 0.2375,
        "free_cash_flow_margin": 0.2375,
    },
    summary=(
        "The company has strong financial fundamentals "
        "but faces elevated business and regulatory risks."
    ),
    data_status=True,
)


# 4. ValuationAgent mock output
valuation_output = ValuationAnalysisOutput(
    ticker="APPL",
    valuation_status="SLIGHTLY UNDERVALUED",

    dcf_valuation=DCFValuation(
        enterprise_value=1800000000000,
        equity_value=1700000000000,
        intrinsic_value_per_share=113.33,
        current_market_price=100.00,
        upside_percentage=13.33,
    ),

    peer_valuation=PeerValuation(
        peer_tickers=["MSFT", "GOOGL", "AMZN"],

        company_multiples=ValuationMultiples(
            pe_ratio=25.0,
            ev_ebitda=16.0,
            ps_ratio=4.5,
        ),

        peer_average_multiples=ValuationMultiples(
            pe_ratio=28.0,
            ev_ebitda=18.0,
            ps_ratio=5.0,
        ),

        relative_valuation="Trading at a discount",
        summary=(
            "The stock trades at lower valuation "
            "multiples than the selected peer group."
        ),
    ),

    key_findings=[
        "DCF estimates 13.33% potential upside",
        "Stock trades at a discount to selected peers",
    ],

    summary=(
        "The stock appears slightly undervalued based "
        "on the mock DCF estimate and peer comparison."
    ),

    data_status=True,
)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    try:
        # Mock specialist outputs prepared for testing
        # Replace these with actual outputs during orchestration

        report = generate_investment_report(
            financial_data=financial_output,
            news_data=news_output,
            risk_data=risk_output,
            valuation_data=valuation_output,
        )

        print(report.model_dump_json(indent=2))

    except Exception:
        logger.exception("ReportWriterAgent failed.")
        raise