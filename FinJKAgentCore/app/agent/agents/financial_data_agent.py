import logging
import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from strands import Agent
from strands.models.openai import OpenAIModel

from tools.financial_tools import (
    compute_key_ratios,
    get_balance_sheet,
    get_cash_flow_statement,
    get_income_statement,
)

load_dotenv()  # reads GROQ_API_KEY from a .env file, if present

logger=logging.getLogger(__name__)

MODEL_ID = os.getenv("GROQ_MODEL_ID", "openai/gpt-oss-120b")

class FinancialDataOutput(BaseModel):
    ticker:str
    overall_company_health:Literal[
        "STRONG",
        "WEAK",
        "MODERATE",
        "MIXED"
    ]
    period:Literal[
        "annual",
        "quarterly"
    ]
    #Income statement
    revenue:float|None
    cost_of_revenue:float
    operating_income:float
    net_income:float|None
    eps:float |None=None
    #Balance sheet
    total_assets:float|None
    total_liabilities:float
    shareholders_equity:float
    cash_and_equivalents:float
    total_debt:float
    #Cash Flow statement
    operating_cash_flow:float
    investing_cash_flow:float
    financing_cash_flow:float
    capital_expenditure:float
    free_cash_flow:float
    #Financial Ratios
    net_profit_margin:float|None = None
    return_on_equity:float|None = None
    debt_to_equity:float|None = None
    free_cash_flow_margin:float|None = None
    #Data avaialbility
    data_status:bool
    #Overall Financial Assesment
    summary:str=Field(
        description="Concise explanation of the company's financial helath based on financial statements and key ratios."
    )


SYSTEM_PROMPT="""

ROLE: You are a FinancialData, a financial data retrieval and analysis specialist.

OBJECTIVE: Retrieve a company's financial statements, obtain computed financial ratios, and provide an evidence-based assessment of the company's financial health.

INPUTS: You will receive a company's stock ticker as a string.
Use the uppercase ticker consistently.

TOOLS: 
1. get_income_statement:
   Retrieve revenue, expenses, operating income,
   net income, and EPS.

2. get_balance_sheet:
   Retrieve assets, liabilities, shareholders' equity,
   cash balances, and debt.

3. get_cash_flow_statement:
   Retrieve operating, investing, and financing
   cash flows, capital expenditure, and free cash flow.

4. compute_key_ratios:
   Retrieve calculated profitability, leverage,
   and cash flow ratios.

EXECUTION RULES:
- Use the available financial tools to retrieve data.
- Ensure all retrieved statements correspond to the
  requested company.
- Preserve financial figures exactly as returned by
  the tools.
- Use the computed ratios returned by
  compute_key_ratios.
- Do not independently invent or recalculate financial
  figures.
- Do not assume unavailable values are zero.
- Do not use external information to fill missing fields.

FINANCIAL HEALTH ASSESSMENT:
Based on profitability, leverage, and cash generation,
classify the company's financial health as:

    STRONG:
        Healthy profitability, manageable debt, and strong
        cash generation.

    MODERATE:
        Generally stable financial performance with some
        financial weaknesses.

    WEAK:
        Significant financial weaknesses, such as persistent
        losses, excessive debt, or poor cash generation.

    MIXED:
        Conflicting financial indicators, such as strong
        profitability combined with substantial debt.

    These classifications are qualitative assessments,
    not fixed numerical thresholds.

CONSTRAINTS:
- Use only the registered financial tools.
- Do not fabricate financial information.
- Do not confuse financial health with stock valuation.
- Do not provide buy, sell, or hold recommendations.
- If required data is unavailable, clearly indicate
  the limitation.
- If financial values conflict, do not silently
  correct them or conceal the inconsistency.
- Do not present mock financial information as live data.

OUTPUT:
    Return a structured FinancialDataOutput containing
    the retrieved financial metrics, calculated ratios,
    overall company health classification, concise summary,
    and data availability status.

    The summary should explain the main financial
    strengths and weaknesses based on the retrieved data.
"""


def create_financial_data_agent()->Agent:
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
    logger.info("Initializing FinancialDataAgent with model: %s", MODEL_ID)

    return Agent(
        name="FinancialDataAgent",
        model=model,
        # callback_handler=None,
        system_prompt=SYSTEM_PROMPT,
        tools=[get_income_statement, get_balance_sheet, get_cash_flow_statement, compute_key_ratios],
        structured_output_model=FinancialDataOutput
    )


if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)

    agent=create_financial_data_agent()

    try:
        result=agent("Retrieve and Analyze Financial data for ticker AAPL.")

        output=result.structured_output

        if output is None:
            raise ValueError("Agent returned no structured output")

        print(output.model_dump_json(indent=2))

    except Exception:
        logger.exception("Financial Data Agent failed.")
        raise