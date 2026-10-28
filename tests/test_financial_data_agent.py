
from unittest.mock import MagicMock, patch

import pytest

from agents.financial_data_agent import (
    FinancialDataOutput,
    create_financial_data_agent,
)
from tools.financial_tools import (
    compute_key_ratios,
    get_balance_sheet,
    get_cash_flow_statement,
    get_income_statement,
)


@pytest.fixture
def mock_financial_output():
    """Return a valid mocked financial agent response."""

    return FinancialDataOutput(
        ticker="AAPL",
        overall_company_health="STRONG",
        period="annual",

        # Income Statement
        revenue=40000000,
        cost_of_revenue=22000000,
        operating_income=12000000,
        net_income=9500000,
        eps=6.25,

        # Balance Sheet
        total_assets=35000000,
        total_liabilities=28000000,
        shareholders_equity=7000000,
        cash_and_equivalents=5000000,
        total_debt=1000000,

        # Cash Flow
        operating_cash_flow=11000000,
        investing_cash_flow=-2000000,
        financing_cash_flow=-7000000,
        capital_expenditure=1500000,
        free_cash_flow=9500000,

        # Financial Ratios
        net_profit_margin=0.2375,
        return_on_equity=1.3571,
        debt_to_equity=0.1429,
        free_cash_flow_margin=0.2375,

        data_status=True,

        summary=(
            "The company demonstrates healthy profitability, "
            "manageable leverage, and strong cash generation."
        ),
    )


# TEST 1: Agent Creation
@patch("agents.financial_data_agent.Agent")
@patch("agents.financial_data_agent.OpenAIModel")
def test_create_financial_data_agent(
    mock_model,
    mock_agent,
    monkeypatch,
):
    monkeypatch.setenv("GROQ_API_KEY", "fake-test-key")

    agent = create_financial_data_agent()

    mock_model.assert_called_once()
    mock_agent.assert_called_once()

    assert agent is mock_agent.return_value


# TEST 2: Tool Registration
@patch("agents.financial_data_agent.Agent")
@patch("agents.financial_data_agent.OpenAIModel")
def test_financial_agent_tools(
    mock_model,
    mock_agent,
    monkeypatch,
):
    monkeypatch.setenv("GROQ_API_KEY", "fake-test-key")

    create_financial_data_agent()

    kwargs = mock_agent.call_args.kwargs

    assert kwargs["name"] == "FinancialDataAgent"

    assert kwargs["tools"] == [
        get_income_statement,
        get_balance_sheet,
        get_cash_flow_statement,
        compute_key_ratios,
    ]

    assert kwargs["structured_output_model"] is FinancialDataOutput


# TEST 3: Mock Agent Response
@patch("agents.financial_data_agent.Agent")
@patch("agents.financial_data_agent.OpenAIModel")
def test_financial_agent_response(
    mock_model,
    mock_agent,
    mock_financial_output,
    monkeypatch,
):
    monkeypatch.setenv("GROQ_API_KEY", "fake-test-key")

    # Simulate an AgentResult
    fake_result = MagicMock()
    fake_result.structured_output = mock_financial_output

    fake_agent = mock_agent.return_value
    fake_agent.return_value = fake_result

    agent = create_financial_data_agent()

    result = agent("Analyze financial data for APPL.")

    output = result.structured_output

    assert isinstance(output, FinancialDataOutput)
    assert output.ticker == "AAPL"
    assert output.overall_company_health == "STRONG"
    assert output.data_status is True

    assert output.revenue == 40000000
    assert output.net_income == 9500000
    assert output.free_cash_flow == 9500000

    assert output.net_profit_margin == pytest.approx(0.2375)
    assert output.debt_to_equity == pytest.approx(0.1429)

    fake_agent.assert_called_once_with(
        "Analyze financial data for APPL."
    )


# TEST 4: Invalid Financial Health Classification
def test_invalid_company_health(mock_financial_output):
    data = mock_financial_output.model_dump()

    data["overall_company_health"] = "EXCELLENT"

    with pytest.raises(ValueError):
        FinancialDataOutput(**data)


# TEST 5: Invalid Data Types
def test_invalid_revenue_type(mock_financial_output):
    data = mock_financial_output.model_dump()

    data["revenue"] = "not a number"

    with pytest.raises(ValueError):
        FinancialDataOutput(**data)


# TEST 6: Nullable Financial Ratio
def test_nullable_financial_ratio(mock_financial_output):
    data = mock_financial_output.model_dump()

    data["return_on_equity"] = None

    result = FinancialDataOutput(**data)

    assert result.return_on_equity is None
