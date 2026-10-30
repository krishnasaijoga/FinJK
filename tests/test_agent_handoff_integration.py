import json
import os

import pytest
from dotenv import load_dotenv
from strands.hooks import BeforeToolCallEvent

from agents.valuation_agent import (
    create_valuation_analysis_agent,
    ValuationAnalysisOutput,
)


load_dotenv()


@pytest.mark.integration
def test_valuation_agent_passes_financial_data_to_dcf(
    mock_yahoo,
    financial_agent_output,
):
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not configured")

    captured_dcf_calls = []

    # Create real ValuationAgent with existing tools.
    agent = create_valuation_analysis_agent()

    # Capture the arguments supplied by the LLM
    # immediately before the DCF tool executes.
    def capture_tool_call(event: BeforeToolCallEvent):
        tool_use = event.tool_use

        if tool_use["name"] == "run_dcf_model":
            captured_dcf_calls.append(
                dict(tool_use["input"])
            )

    agent.add_hook(
        capture_tool_call,
        BeforeToolCallEvent,
    )

    # Simulate financial data arriving from
    # FinancialDataAgent.
    prompt = f"""
    Analyze ticker AAPL.

    Below is the FinancialDataAgent structured output.

    {json.dumps(financial_agent_output, indent=2)}

    Instructions:
    - Use the supplied financial metrics.
    - Pass them into run_dcf_model using financial_data.
    - Preserve every numerical value exactly.
    - Do not invent missing financial values.
    - The financial figures are synthetic test data.
    - Do not represent these figures as verified
      Apple financial information.

    Return a structured ValuationAnalysisOutput.
    """

    # This invokes Groq and the registered tools.
    result = agent(prompt)

    # Verify the tool was called.
    assert captured_dcf_calls, (
        "ValuationAgent did not call run_dcf_model"
    )

    # Verify financial values passed into DCF.
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

        assert "financial_data" in tool_input, (
            "DCF call missing financial_data argument"
        )

        received_data = tool_input["financial_data"]

        for field in expected_fields:
            assert received_data[field] == (
                financial_agent_output[field]
            ), (
                f"Mismatch in {field}: "
                f"expected {financial_agent_output[field]}, "
                f"received {received_data[field]}"
            )

    # Verify structured output.
    assert result.structured_output is not None

    output = ValuationAnalysisOutput.model_validate(
        result.structured_output
    )

    assert output.ticker == "AAPL"

    print(
        "\nCaptured DCF arguments:",
        json.dumps(captured_dcf_calls, indent=2),
    )

    print(
        "\nValuationAgent output:",
        output.model_dump_json(indent=2),
    )
