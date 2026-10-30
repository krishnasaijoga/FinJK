
from unittest.mock import MagicMock, patch

import pandas as pd

from providers.yahoo_finance import get_metric, get_statement


def test_income_statement_mapping():
    mock_stock = MagicMock()

    mock_stock.get_income_stmt.return_value = pd.DataFrame(
        {
            pd.Timestamp("2025-09-30"): {
                "Total Revenue": 40000000,
                "Cost Of Revenue": 22000000,
                "Operating Income": 12000000,
                "Net Income": 9500000,
                "Diluted EPS": 6.25,
            }
        }
    )

    with patch(
        "providers.yahoo_finance.get_stock",
        return_value=mock_stock,
    ):
        result = get_statement(
            "AAPL", "income", "annual"
        )

    assert result["data_status"] is True
    assert result["report_date"] == "2025-09-30"
    assert result["metrics"]["Total Revenue"] == 40000000

    mock_stock.get_income_stmt.assert_called_once_with(
        freq="yearly"
    )


def test_missing_metric_returns_none():
    assert get_metric(
        {"Total Revenue": 100},
        "Operating Income",
        "OperatingIncome",
    ) is None


def test_empty_statement():
    mock_stock = MagicMock()
    mock_stock.get_income_stmt.return_value = pd.DataFrame()

    with patch(
        "providers.yahoo_finance.get_stock",
        return_value=mock_stock,
    ):
        result = get_statement(
            "AAPL", "income", "annual"
        )

    assert result["data_status"] is False
    assert result["metrics"] == {}
