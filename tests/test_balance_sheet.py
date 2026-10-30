
from unittest.mock import patch

import pytest

# Change this import if your tool is in a different module
from tools.financial_tools import get_balance_sheet


@pytest.fixture
def mock_balance_data():
    return {
        "ticker": "AAPL",
        "period": "annual",
        "report_date": "2025-09-30",
        "data_status": True,
        "metrics": {
            "Total Assets": 35000000,
            "Total Liabilities Net Minority Interest": 28000000,
            "Stockholders Equity": 7000000,
            "Cash And Cash Equivalents": 5000000,
            "Total Debt": 1000000,
        },
    }


# 1. Successful balance sheet retrieval
def test_get_balance_sheet_success(mock_balance_data):
    with patch(
        "tools.financial_tools.get_statement",
        return_value=mock_balance_data,
    ) as mock_get:
        result = get_balance_sheet("AAPL")

    mock_get.assert_called_once_with(
        "AAPL", "balance", "annual"
    )

    assert result["ticker"] == "AAPL"
    assert result["period"] == "annual"
    assert result["report_date"] == "2025-09-30"
    assert result["total_assets"] == 35000000
    assert result["total_liabilities"] == 28000000
    assert result["shareholders_equity"] == 7000000
    assert result["cash_and_equivalents"] == 5000000
    assert result["total_debt"] == 1000000
    assert result["data_status"] is True
    assert result["source"] == "Yahoo Finance"


# 2. Quarterly data
def test_get_balance_sheet_quarterly(mock_balance_data):
    quarterly_data = {
        **mock_balance_data,
        "period": "quarterly",
        "report_date": "2026-06-30",
    }

    with patch(
        "tools.financial_tools.get_statement",
        return_value=quarterly_data,
    ) as mock_get:
        result = get_balance_sheet(
            "AAPL", period="quarterly"
        )

    mock_get.assert_called_once_with(
        "AAPL", "balance", "quarterly"
    )
    assert result["period"] == "quarterly"
    assert result["report_date"] == "2026-06-30"


# 3. Alternate Yahoo Finance field names
def test_get_balance_sheet_fallback_keys(mock_balance_data):
    fallback_data = {
        **mock_balance_data,
        "metrics": {
            "TotalAssets": 35000000,
            "TotalLiabilitiesNetMinorityInterest": 28000000,
            "CommonStockEquity": 7000000,
            "CashAndCashEquivalents": 5000000,
            "TotalDebt": 1000000,
        },
    }

    with patch(
        "tools.financial_tools.get_statement",
        return_value=fallback_data,
    ):
        result = get_balance_sheet("AAPL")

    assert result["total_assets"] == 35000000
    assert result["total_liabilities"] == 28000000
    assert result["shareholders_equity"] == 7000000
    assert result["cash_and_equivalents"] == 5000000
    assert result["total_debt"] == 1000000
    assert result["data_status"] is True


# 4. Missing one required metric
def test_get_balance_sheet_missing_metric(mock_balance_data):
    incomplete_data = {
        **mock_balance_data,
        "metrics": {
            key: value
            for key, value in mock_balance_data["metrics"].items()
            if key != "Total Debt"
        },
    }

    with patch(
        "tools.financial_tools.get_statement",
        return_value=incomplete_data,
    ):
        result = get_balance_sheet("AAPL")

    assert result["total_debt"] is None
    assert result["data_status"] is False

    # Available metrics should still be preserved.
    assert result["total_assets"] == 35000000


# 5. Yahoo Finance returns an empty statement
def test_get_balance_sheet_no_data():
    empty_data = {
        "ticker": "INVALID",
        "period": "annual",
        "report_date": None,
        "data_status": False,
        "metrics": {},
    }

    with patch(
        "tools.financial_tools.get_statement",
        return_value=empty_data,
    ):
        result = get_balance_sheet("INVALID")

    assert result["data_status"] is False
    assert result["metrics"] == {}
    assert result["report_date"] is None


# 6. All metrics missing
def test_get_balance_sheet_all_metrics_missing(mock_balance_data):
    missing_data = {
        **mock_balance_data,
        "metrics": {},
    }

    with patch(
        "tools.financial_tools.get_statement",
        return_value=missing_data,
    ):
        result = get_balance_sheet("AAPL")

    assert result["data_status"] is False
    assert result["total_assets"] is None
    assert result["total_liabilities"] is None
    assert result["shareholders_equity"] is None
    assert result["cash_and_equivalents"] is None
    assert result["total_debt"] is None


# 7. Provider error should propagate
def test_get_balance_sheet_provider_error():
    with patch(
        "tools.financial_tools.get_statement",
        side_effect=ConnectionError("Yahoo Finance unavailable"),
    ), pytest.raises(
        ConnectionError,
        match="Yahoo Finance unavailable",
    ):
        get_balance_sheet("AAPL")


# 8. Do not confuse cash with short-term investments
def test_cash_does_not_include_short_term_investments(
    mock_balance_data,
):
    data = {
        **mock_balance_data,
        "metrics": {
            **mock_balance_data["metrics"],
            "Cash Cash Equivalents And Short Term Investments": 9000000,
        },
    }

    with patch(
        "tools.financial_tools.get_statement",
        return_value=data,
    ):
        result = get_balance_sheet("AAPL")

    assert result["cash_and_equivalents"] == 5000000
    assert result["data_status"] is True
