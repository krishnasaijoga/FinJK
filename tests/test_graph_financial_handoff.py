
import json
import os

import pytest

from dotenv import load_dotenv
from strands import Agent
from strands.hooks import BeforeToolCallEvent
from strands.multiagent import GraphBuilder
from strands.multiagent.base import Status

from agents.valuation_agent import (
    create_valuation_analysis_agent,
    ValuationAnalysisOutput,
)


load_dotenv()


@pytest.mark.integration
def test_graph_passes_financial_data_to_valuation(
    mock_yahoo,
    financial_agent_output,
):
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not configured")

    captured_dcf_calls = []

    # ==========================================
    # BLOCK 1: Create real ValuationAgent
    # ==========================================

    valuation_agent = create_valuation_analysis_agent()

    # ==========================================
    # BLOCK 2: Create controlled upstream agent
    # ==========================================

    financial_agent = Agent(
        name="FinancialDataAgent",
        model=valuation_agent.model,
        tools=[],
        system_prompt=f"""
        You are a controlled financial-data agent
        for an integration test.

        When asked to analyze AAPL, return ONLY
        the following JSON object, without
        markdown, explanations, or modifications:

        {json.dumps(financial_agent_output)}

        All figures are synthetic test inputs.
        Do not fetch or calculate financial data.
        """,
    )

    # ==========================================
    # BLOCK 3: Capture ValuationAgent tool calls
    # ==========================================

    def capture_tool_call(event: BeforeToolCallEvent):
        tool_use = event.tool_use

        if tool_use["name"] == "run_dcf_model":
            captured_dcf_calls.append(
                dict(tool_use["input"])
            )

    valuation_agent.add_hook(
        capture_tool_call,
        BeforeToolCallEvent,
    )

    # ==========================================
    # BLOCK 4: Build a real two-node Graph
    # ==========================================

    builder = GraphBuilder()

    builder.add_node(
        financial_agent,
        "financial",
    )

    builder.add_node(
        valuation_agent,
        "valuation",
    )

    builder.set_entry_point("financial")

    builder.add_edge(
        "financial",
        "valuation",
    )

    builder.set_max_node_executions(2)

    graph = builder.build()

    # ==========================================
    # BLOCK 5: Execute the Graph
    # ==========================================

    result = graph(
        "Analyze stock ticker AAPL. "
        "Use the financial agent's output for DCF. "
        "All financial figures are synthetic."
    )

    # ==========================================
    # BLOCK 6: Verify Graph execution
    # ==========================================

    assert result.status == Status.COMPLETED, result.status

    assert "financial" in result.results
    assert "valuation" in result.results

    assert (
        result.results["financial"].status
        == Status.COMPLETED
    )

    assert (
        result.results["valuation"].status
        == Status.COMPLETED
    )

    # ==========================================
    # BLOCK 7: Verify upstream agent output
    # ==========================================

    financial_result = result.results["financial"].result

    assert financial_result is not None

    # The controlled upstream agent should have
    # emitted the supplied financial figures.
    financial_response = str(financial_result)

    for field in [
        "operating_income",
        "cash_and_equivalents",
        "total_debt",
        "capital_expenditure",
    ]:
        assert field in financial_response

    # ==========================================
    # BLOCK 8: Verify DCF handoff
    # ==========================================

    assert captured_dcf_calls, (
        "ValuationAgent never invoked run_dcf_model"
    )

    expected_fields = [
        "ticker",
        "data_status",
        "operating_income",
        "cash_and_equivalents",
        "total_debt",
        "capital_expenditure",
    ]

    for tool_input in captured_dcf_calls:
        assert tool_input["ticker"].strip().upper() == "AAPL"

        assert "financial_data" in tool_input

        received = tool_input["financial_data"]

        for field in expected_fields:
            assert received[field] == (
                financial_agent_output[field]
            ), (
                f"Graph handoff mismatch: {field}. "
                f"Expected {financial_agent_output[field]}, "
                f"received {received[field]}"
            )

    # ==========================================
    # BLOCK 9: Verify valuation output
    # ==========================================

    valuation_result = result.results["valuation"].result

    assert valuation_result is not None

    assert valuation_result.structured_output is not None # type: ignore

    output = ValuationAnalysisOutput.model_validate(
        valuation_result.structured_output # type: ignore
    )

    assert output.ticker == "AAPL"

    # ==========================================
    # BLOCK 10: Debug information
    # ==========================================

    print("\nGraph nodes:")
    print(list(result.results.keys()))

    print("\nUpstream financial response:")
    print(financial_response)

    print("\nCaptured DCF tool arguments:")
    print(json.dumps(captured_dcf_calls, indent=2))

    print("\nValuation output:")
    print(output.model_dump_json(indent=2))
