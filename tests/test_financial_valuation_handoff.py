
from unittest.mock import patch

from tools.valuation_tools import run_dcf_model


def test_dcf_rejects_mismatched_financial_ticker():
    financial_data = {
        "ticker": "NVDA",
        "data_status": True,
        "operating_income": 100_000_000,
        "cash_and_equivalents": 10_000_000,
        "total_debt": 20_000_000,
        "capital_expenditure": -15_000_000,
    }

    with patch(
        "tools.valuation_tools.get_statement"
    ) as mock_statement:
        result = run_dcf_model(
            ticker="AAPL",
            financial_data=financial_data,
        )

    assert result["data_status"] is False
    assert result["error"] == "Financial data ticker mismatch"

    mock_statement.assert_not_called()


def test_dcf_rejects_incomplete_financial_data():
    financial_data = {
        "ticker": "AAPL",
        "data_status": False,
    }

    with patch(
        "tools.valuation_tools.get_statement"
    ) as mock_statement:
        result = run_dcf_model(
            ticker="AAPL",
            financial_data=financial_data,
        )

    assert result["data_status"] is False
    assert "insufficient data" in result["error"].lower()

    mock_statement.assert_not_called()
