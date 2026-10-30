
from unittest.mock import patch

import pytest

from tools.financial_tools import get_cash_flow_statement


@pytest.fixture
def mock_cash_flow_data():
    return {
        "ticker": "AAPL",
        "period": "annual",
        "report_date": "2025-09-30",
        "data_status": True,
        "metrics": {
            "Operating Cash Flow": 11_000_000,
            "Investing Cash Flow": -2_000_000,
            "Financing Cash Flow": -7_000_000,
            "Capital Expenditure": -1_500_000,
            "Free Cash Flow": 9_500_000,
        },
    }


def test_cash_flow_success(mock_cash_flow_data):
    with patch(
        "tools.financial_tools.get_statement",
        return_value=mock_cash_flow_data,
    ) as mock_get:
        result = get_cash_flow_statement("AAPL")

    mock_get.assert_called_once_with(
        "AAPL", "cashflow", "annual"
    )

    assert result["ticker"] == "AAPL"
    assert result["operating_cash_flow"] == 11_000_000
    assert result["investing_cash_flow"] == -2_000_000
    assert result["financing_cash_flow"] == -7_000_000
    assert result["capital_expenditure"] == -1_500_000
    assert result["free_cash_flow"] == 9_500_000
    assert result["data_status"] is True
    assert result["source"] == "Yahoo Finance"


def test_cash_flow_quarterly(mock_cash_flow_data):
    data = {
        **mock_cash_flow_data,
        "period": "quarterly",
        "report_date": "2026-06-30",
    }

    with patch(
        "tools.financial_tools.get_statement",
        return_value=data,
    ):
        result = get_cash_flow_statement(
            "AAPL", period="quarterly"
        )

    assert result["period"] == "quarterly"
    assert result["report_date"] == "2026-06-30"


def test_cash_flow_fallback_keys(mock_cash_flow_data):
    data = {
        **mock_cash_flow_data,
        "metrics": {
            "OperatingCashFlow": 11_000_000,
            "InvestingCashFlow": -2_000_000,
            "FinancingCashFlow": -7_000_000,
            "CapitalExpenditure": -1_500_000,
            "FreeCashFlow": 9_500_000,
        },
    }

    with patch(
        "tools.financial_tools.get_statement",
        return_value=data,
    ):
        result = get_cash_flow_statement("AAPL")

    assert result["operating_cash_flow"] == 11_000_000
    assert result["investing_cash_flow"] == -2_000_000
    assert result["financing_cash_flow"] == -7_000_000
    assert result["capital_expenditure"] == -1_500_000
    assert result["free_cash_flow"] == 9_500_000
    assert result["data_status"] is True


def test_fcf_calculated_when_missing(mock_cash_flow_data):
    data = {
        **mock_cash_flow_data,
        "metrics": {
            key: value
            for key, value in mock_cash_flow_data["metrics"].items()
            if key != "Free Cash Flow"
        },
    }

    with patch(
        "tools.financial_tools.get_statement",
        return_value=data,
    ):
        result = get_cash_flow_statement("AAPL")

    assert result["free_cash_flow"] == 9_500_000
    assert result["data_status"] is True


def test_missing_capex_without_fcf(mock_cash_flow_data):
    data = {
        **mock_cash_flow_data,
        "metrics": {
            key: value
            for key, value in mock_cash_flow_data["metrics"].items()
            if key not in ("Capital Expenditure", "Free Cash Flow")
        },
    }

    with patch(
        "tools.financial_tools.get_statement",
        return_value=data,
    ):
        result = get_cash_flow_statement("AAPL")

    assert result["capital_expenditure"] is None
    assert result["free_cash_flow"] is None
    assert result["data_status"] is False


def test_cash_flow_no_data():
    data = {
        "ticker": "INVALID",
        "period": "annual",
        "report_date": None,
        "data_status": False,
        "metrics": {},
    }

    with patch(
        "tools.financial_tools.get_statement",
        return_value=data,
    ):
        result = get_cash_flow_statement("INVALID")

    assert result["data_status"] is False
    assert result["metrics"] == {}


def test_cash_flow_provider_failure():
    with patch(
        "tools.financial_tools.get_statement",
        side_effect=ConnectionError("Yahoo Finance unavailable"),
    ), pytest.raises(
        ConnectionError,
        match="Yahoo Finance unavailable",
    ):
        get_cash_flow_statement("AAPL")


def test_fcf_uses_signed_capex(mock_cash_flow_data):
    data = {
        **mock_cash_flow_data,
        "metrics": {
            **mock_cash_flow_data["metrics"],
            "Free Cash Flow": None,
        },
    }

    with patch(
        "tools.financial_tools.get_statement",
        return_value=data,
    ):
        result = get_cash_flow_statement("AAPL")

    assert result["free_cash_flow"] == (
        result["operating_cash_flow"]
        + result["capital_expenditure"]
    )
