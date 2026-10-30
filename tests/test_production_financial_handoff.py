import os
from unittest.mock import patch

import pytest
from dotenv import load_dotenv
from strands.hooks import BeforeToolCallEvent
from strands.multiagent import GraphBuilder
from strands.multiagent.base import Status

from agents.financial_data_agent import create_financial_data_agent
from agents.valuation_agent import create_valuation_analysis_agent


load_dotenv()


@pytest.mark.integration
def test_production_financial_to_valuation_handoff(
    mock_yahoo,
):
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not configured")

    captured_dcf_calls = []

    financial_agent = create_financial_data_agent()
    valuation_agent = create_valuation_analysis_agent()

    # Capture financial data supplied to DCF.
    def capture_dcf(event: BeforeToolCallEvent):
        if event.tool_use["name"] == "run_dcf_model":
            captured_dcf_calls.append(
                dict(event.tool_use["input"])
            )

    valuation_agent.add_hook(
        capture_dcf,
        BeforeToolCallEvent,
    )

    builder = GraphBuilder()

    builder.add_node(financial_agent, "financial")
    builder.add_node(valuation_agent, "valuation")

    builder.set_entry_point("financial")
    builder.add_edge("financial", "valuation")
    builder.set_max_node_executions(2)

    graph = builder.build()

    result = graph(
        "Analyze stock ticker AAPL. "
        "Use FinancialDataAgent's output for DCF."
    )

    assert result.status == Status.COMPLETED

    financial_result = result.results["financial"].result
    valuation_result = result.results["valuation"].result

    assert financial_result is not None
    assert valuation_result is not None

    assert financial_result.structured_output is not None # type: ignore
    assert valuation_result.structured_output is not None # type: ignore

    financial_output = financial_result.structured_output # type: ignore

    assert captured_dcf_calls, (
        "ValuationAgent did not invoke run_dcf_model"
    )

    for tool_input in captured_dcf_calls:
        assert tool_input["ticker"].upper() == "AAPL"

        received = tool_input["financial_data"]

        for field in (
            "operating_income",
            "cash_and_equivalents",
            "total_debt",
            "capital_expenditure",
        ):
            assert received[field] == getattr(
                financial_output, field
            ), f"Mismatch in {field}"
