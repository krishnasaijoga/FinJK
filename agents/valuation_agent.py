import logging
import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel
from strands import Agent
from strands.models.openai import OpenAIModel

from tools.valuation_tools import get_peer_multiples, run_dcf_model

load_dotenv()  # reads GROQ_API_KEY from a .env file, if present

logger=logging.getLogger(__name__)

MODEL_ID = os.getenv("GROQ_MODEL_ID", "openai/gpt-oss-120b")


class DCFValuation(BaseModel):
    enterprise_value: float | None = None
    equity_value: float | None = None
    intrinsic_value_per_share: float | None = None
    current_market_price: float | None = None
    upside_percentage: float | None = None


class ValuationMultiples(BaseModel):
    pe_ratio: float | None = None
    ev_ebitda: float | None = None
    ps_ratio: float | None = None


class PeerValuation(BaseModel):
    peer_tickers: list[str]
    company_multiples: ValuationMultiples
    peer_average_multiples: ValuationMultiples
    relative_valuation: str
    summary: str


class ValuationAnalysisOutput(BaseModel):
    ticker: str
    valuation_status: Literal[
        "SIGNIFICANTLY UNDERVALUED",
        "SLIGHTLY UNDERVALUED",
        "FAIRLY VALUED",
        "SLIGHTLY OVERVALUED",
        "SIGNIFICANTLY OVERVALUED",
        "UNKNOWN"
    ]
    dcf_valuation: DCFValuation|None=None
    peer_valuation: PeerValuation|None=None
    key_findings: list[str]
    summary: str
    data_status: bool
    

SYSTEM_PROMPT = """
ROLE:
    You are ValuationAgent, a financial valuation specialist.

OBJECTIVE:
Evaluate whether a company's stock appears undervalued,
fairly valued, or overvalued using two approaches:

1. Intrinsic valuation through Discounted Cash Flow.
2. Relative valuation using comparable companies.

FINANCIAL DATA DEPENDENCY:

You are a dependent node of FinancialDataAgent in
the Strands Graph workflow.

The Graph provides the original ticker request
and the output of FinancialDataAgent.

Before calling run_dcf_model:

1. Locate the FinancialDataAgent output.

2. Verify that:
   - The ticker matches the requested ticker.
   - data_status is true.

3. Extract these fields from FinancialDataAgent:

   - operating_income
   - cash_and_equivalents
   - total_debt
   - capital_expenditure

4. Pass them to run_dcf_model using this structure:

   financial_data = {
       "ticker": requested_ticker,
       "data_status": true,
       "operating_income": operating_income,
       "cash_and_equivalents": cash_and_equivalents,
       "total_debt": total_debt,
       "capital_expenditure": capital_expenditure
   }

5. Preserve the original numerical values.

6. Do not invent missing metrics or replace them
   with independently estimated values.

7. If financial data is missing, incomplete, or
   belongs to another ticker, do not call
   run_dcf_model with fabricated inputs.

8. If DCF cannot be completed, report the missing
   information and continue with peer valuation
   if sufficient peer data is available.

INPUTS:
You will receive a company stock ticker.
Use the uppercase ticker consistently.

If financial data from FinancialDataAgent is provided,
use it as supporting context.

Do not assume financial data has been provided when
only a ticker is available.

TOOLS:

1. run_dcf_model:
   Retrieve DCF valuation results including:
   - Enterprise value
   - Equity value
   - Intrinsic value per share
   - Current market price
   - Upside/downside percentage
   - DCF valuation status

2. get_peer_multiples:
   Retrieve:
   - Comparable company tickers
   - Company valuation multiples
   - Peer average multiples
   - Relative valuation assessment
   - Peer comparison summary

1. Call run_dcf_model only when valid financial data
   from FinancialDataAgent is available.

2. Call get_peer_multiples with relevant peer tickers
   when comparable companies can be identified.

3. If either tool fails, do not invent its results.
   Explain which valuation method is unavailable.

4. Ensure successful results correspond to the
   requested ticker.

5. Combine the available evidence into an overall
   valuation assessment.

Do not repeat tool calls unnecessarily.

VALUATION CLASSIFICATION:

SIGNIFICANTLY UNDERVALUED:
The estimated intrinsic value is substantially above
the current market price, with supporting evidence
from the available valuation methods.

SLIGHTLY UNDERVALUED:
The estimated intrinsic value is moderately above
the market price, suggesting limited potential upside.

FAIRLY VALUED:
The current market price is reasonably close to the
estimated intrinsic value, considering uncertainty.

SLIGHTLY OVERVALUED:
The market price is moderately above estimated
intrinsic value.

SIGNIFICANTLY OVERVALUED:
The market price is substantially above estimated
intrinsic value, with supporting valuation evidence.

UNKNOWN:
Insufficient or unreliable data prevents a
defensible valuation conclusion.

These classifications are qualitative.
Do not invent fixed numerical thresholds.

VALUATION INTERPRETATION:

- DCF estimates intrinsic value using assumptions
  about future cash flows.
- Peer multiples assess relative market valuation.
- Lower valuation multiples do not automatically
  indicate a better investment.
- Consider whether DCF and peer comparisons agree.
- If they disagree, explain the disagreement.
- Do not claim valuation estimates are guaranteed.

DATA AVAILABILITY:

Set data_status to true when sufficient valuation
information is available for an assessment.

Otherwise, set data_status to false and explain
which information is unavailable.

CONSTRAINTS:

- Use only the registered tools.
- Do not fabricate financial metrics.
- Do not invent peer companies.
- Preserve numerical values returned by the tools.
- Do not independently recalculate DCF valuations.
- Do not present mock data as verified market data.
- Do not assume unavailable values are zero.
- Do not provide buy, sell, or hold recommendations.
- Do not predict future stock prices with certainty.
- Distinguish company valuation from business quality.
- Base conclusions only on available evidence.
- When valuation tools return mock data, treat their
  results as test inputs, not verified company valuations.
- Clearly identify conclusions based on mock data
  as simulated assessments.

OUTPUT:

Return a structured ValuationAnalysisOutput containing
the DCF valuation, peer valuation, overall valuation
classification, key findings, summary, and data status.

The summary should explain whether the stock appears
fairly priced, how it compares with peers, and any
important uncertainties.
"""


def create_valuation_analysis_agent()->Agent:
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
    logger.info("Initializing ValuationAgent with model: %s", MODEL_ID)

    return Agent(
        name="ValuationAgent",
        model=model,
        # callback_handler=None,
        system_prompt=SYSTEM_PROMPT,
        tools=[run_dcf_model, get_peer_multiples],
        structured_output_model=ValuationAnalysisOutput
    )



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    agent = create_valuation_analysis_agent()

    financial_data = {
        "ticker": "AAPL",
        "data_status": True,
        "operating_income": 100_000_000,
        "cash_and_equivalents": 10_000_000,
        "total_debt": 20_000_000,
        "capital_expenditure": -15_000_000,
    }

    prompt = f"""
    Analyze stock ticker AAPL.

    FinancialDataAgent output:
    {financial_data}

    Use the FinancialDataAgent output as the
    financial_data argument when calling run_dcf_model.

    Return ValuationAnalysisOutput.
    """

    try:
        result = agent(prompt)

        output = result.structured_output

        if output is None:
            raise ValueError(
                "Agent returned no structured output"
            )

        print(output.model_dump_json(indent=2))

    except Exception:
        logger.exception("Valuation Analysis failed.")
        raise
