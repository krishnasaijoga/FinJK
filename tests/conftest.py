
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_financial_data():
    return {
        "income": {
            "ticker": "AAPL",
            "period": "annual",
            "report_date": "2025-09-30",
            "data_status": True,
            "metrics": {
                "EBIT": 100_000_000,
                "Pretax Income": 90_000_000,
                "Tax Provision": 22_500_000,
            },
        },
        "cashflow": {
            "ticker": "AAPL",
            "period": "annual",
            "report_date": "2025-09-30",
            "data_status": True,
            "metrics": {
                "Depreciation And Amortization": 10_000_000,
                "Capital Expenditure": -15_000_000,
                "Change In Working Capital": -5_000_000,
            },
        },
        "balance": {
            "ticker": "AAPL",
            "period": "annual",
            "report_date": "2025-09-30",
            "data_status": True,
            "metrics": {
                "Cash And Cash Equivalents": 10_000_000,
                "Total Debt": 20_000_000,
            },
        },
    }


@pytest.fixture
def mock_yahoo(mock_financial_data):
    def statement_side_effect(ticker, statement, period):
        return mock_financial_data[statement]

    mock_stock = MagicMock()
    mock_stock.info = {
        "sharesOutstanding": 10_000_000,
        "currentPrice": 100.0,
    }

    with (
        patch(
            "tools.valuation_tools.get_statement",
            side_effect=statement_side_effect,
        ) as mock_statement,
        patch(
            "tools.valuation_tools.get_stock",
            return_value=mock_stock,
        ) as mock_get_stock,
    ):
        yield mock_statement, mock_get_stock


@pytest.fixture
def financial_agent_output():
    return {
        "ticker": "AAPL",
        "data_status": True,
        "operating_income": 100_000_000,
        "cash_and_equivalents": 10_000_000,
        "total_debt": 20_000_000,
        "capital_expenditure": -15_000_000,
    }