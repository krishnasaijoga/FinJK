
import pytest
from unittest.mock import patch, MagicMock

from tools.valuation_tools import run_dcf_model


# =====================================================
# FIXTURES
# =====================================================

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


# =====================================================
# 1. SUCCESSFUL DCF CALCULATION
# =====================================================

def test_dcf_success(mock_yahoo, financial_agent_output):
    result = run_dcf_model(
        ticker="AAPL",
        financial_data=financial_agent_output,
    )

    assert result["data_status"] is True, result
    assert result["ticker"] == "AAPL"
    assert result["model"] == "Five-year FCFF DCF"
    assert result["source"] == "Yahoo Finance"

    assert result["base_fcff"] == pytest.approx(
        65_000_000
    )

    assert len(result["forecast"]) == 5
    assert result["enterprise_value"] > 0

    assert result["equity_value"] == pytest.approx(
        result["enterprise_value"] - 10_000_000
    )

    assert result["intrinsic_value_per_share"] == pytest.approx(
        result["equity_value"] / 10_000_000
    )


# =====================================================
# 2. DEFAULT ASSUMPTIONS
# =====================================================

def test_dcf_default_assumptions(
    mock_yahoo,
    financial_agent_output,
):
    result = run_dcf_model(
        ticker="AAPL",
        financial_data=financial_agent_output,
    )

    assert result["data_status"] is True, result

    assert result["assumptions"] == {
        "growth_rate": 0.06,
        "discount_rate": 0.10,
        "terminal_growth_rate": 0.025,
        "forecast_years": 5,
    }


# =====================================================
# 3. CUSTOM ASSUMPTIONS
# =====================================================

def test_dcf_custom_assumptions(
    mock_yahoo,
    financial_agent_output,
):
    result = run_dcf_model(
        ticker="AAPL",
        financial_data=financial_agent_output,
        growth_rate=0.08,
        discount_rate=0.12,
        terminal_growth_rate=0.03,
        forecast_years=3,
    )

    assert result["data_status"] is True, result
    assert len(result["forecast"]) == 3
    assert result["assumptions"]["growth_rate"] == 0.08
    assert result["assumptions"]["discount_rate"] == 0.12
    assert result["assumptions"]["terminal_growth_rate"] == 0.03
    assert result["assumptions"]["forecast_years"] == 3


# =====================================================
# 4. MISSING FINANCIAL STATEMENT
# =====================================================

def test_dcf_missing_statement(
    mock_yahoo,
    mock_financial_data,
    financial_agent_output,
):
    mock_financial_data["cashflow"]["data_status"] = False

    result = run_dcf_model(
        ticker="AAPL",
        financial_data=financial_agent_output,
    )

    assert result["data_status"] is False
    assert "Required financial statements unavailable" in result["error"]


# =====================================================
# 5. MISSING EBIT FROM FINANCIAL AGENT
# =====================================================

def test_dcf_missing_metric(
    mock_yahoo,
    financial_agent_output,
):
    del financial_agent_output["operating_income"]

    result = run_dcf_model(
        ticker="AAPL",
        financial_data=financial_agent_output,
    )

    assert result["data_status"] is False
    assert "ebit" in result["missing_fields"]


# =====================================================
# 6. INVALID PRETAX INCOME
# =====================================================

def test_dcf_invalid_pretax_income(
    mock_yahoo,
    mock_financial_data,
    financial_agent_output,
):
    mock_financial_data["income"]["metrics"][
        "Pretax Income"
    ] = 0

    result = run_dcf_model(
        ticker="AAPL",
        financial_data=financial_agent_output,
    )

    assert result["data_status"] is False
    assert "Cannot derive a valid tax rate" in result["error"]


# =====================================================
# 7. INVALID EFFECTIVE TAX RATE
# =====================================================

def test_dcf_invalid_tax_rate(
    mock_yahoo,
    mock_financial_data,
    financial_agent_output,
):
    mock_financial_data["income"]["metrics"][
        "Tax Provision"
    ] = 100_000_000

    result = run_dcf_model(
        ticker="AAPL",
        financial_data=financial_agent_output,
    )

    assert result["data_status"] is False
    assert "Invalid effective tax rate" in result["error"]


# =====================================================
# 8. MISSING MARKET PRICE
# =====================================================

def test_dcf_missing_market_price(
    mock_yahoo,
    financial_agent_output,
):
    _, mock_get_stock = mock_yahoo

    mock_get_stock.return_value.info = {
        "sharesOutstanding": 10_000_000,
    }

    result = run_dcf_model(
        ticker="AAPL",
        financial_data=financial_agent_output,
    )

    assert result["data_status"] is False
    assert "current_price" in result["missing_fields"]


# =====================================================
# 9. MISSING SHARES OUTSTANDING
# =====================================================

def test_dcf_missing_shares(
    mock_yahoo,
    financial_agent_output,
):
    _, mock_get_stock = mock_yahoo

    mock_get_stock.return_value.info = {
        "currentPrice": 100.0,
    }

    result = run_dcf_model(
        ticker="AAPL",
        financial_data=financial_agent_output,
    )

    assert result["data_status"] is False
    assert "shares_outstanding" in result["missing_fields"]


# =====================================================
# 10. INVALID DISCOUNT RATE
# =====================================================

def test_dcf_invalid_discount_rate(
    mock_yahoo,
    financial_agent_output,
):
    result = run_dcf_model(
        ticker="AAPL",
        financial_data=financial_agent_output,
        discount_rate=0.02,
        terminal_growth_rate=0.03,
    )

    assert result["data_status"] is False
    assert "error" in result


# =====================================================
# 11. NEGATIVE BASE FCFF
# =====================================================

def test_dcf_negative_fcff(
    mock_yahoo,
    financial_agent_output,
):
    # CapEx now comes from FinancialDataAgent,
    # not directly from Yahoo's cashflow statement.
    financial_agent_output[
        "capital_expenditure"
    ] = -200_000_000

    result = run_dcf_model(
        ticker="AAPL",
        financial_data=financial_agent_output,
    )

    assert result["data_status"] is False
    assert "Base FCFF must be positive" in result["error"]


# =====================================================
# 12. INVALID TICKER
# =====================================================

def test_dcf_empty_ticker(financial_agent_output):
    with pytest.raises(
        ValueError,
        match="Ticker cannot be empty",
    ):
        run_dcf_model(
            ticker="   ",
            financial_data=financial_agent_output,
        )


# =====================================================
# 13. TICKER NORMALIZATION
# =====================================================

def test_dcf_ticker_normalization(
    mock_yahoo,
    financial_agent_output,
):
    mock_statement, mock_get_stock = mock_yahoo

    result = run_dcf_model(
        ticker=" aapl ",
        financial_data=financial_agent_output,
    )

    assert result["ticker"] == "AAPL"
    assert result["data_status"] is True, result

    # All statement requests must use the normalized ticker.
    for call in mock_statement.call_args_list:
        assert call.args[0] == "AAPL"

    mock_get_stock.assert_called_once_with("AAPL")


# =====================================================
# 14. REPORT DATES
# =====================================================

def test_dcf_report_dates(
    mock_yahoo,
    financial_agent_output,
):
    result = run_dcf_model(
        ticker="AAPL",
        financial_data=financial_agent_output,
    )

    assert result["data_status"] is True, result

    # The DCF should preserve the dates for the
    # statements it actually uses.
    report_dates = result["report_dates"]

    assert report_dates["income"] == "2025-09-30"
    assert report_dates["cashflow"] == "2025-09-30"

    # Balance sheet may be supplied by FinancialDataAgent.
    if "balance" in report_dates:
        assert report_dates["balance"] == "2025-09-30"


# =====================================================
# 15. UPSIDE CALCULATION
# =====================================================

def test_dcf_upside_calculation(
    mock_yahoo,
    financial_agent_output,
):
    result = run_dcf_model(
        ticker="AAPL",
        financial_data=financial_agent_output,
    )

    assert result["data_status"] is True, result

    intrinsic_value = result["intrinsic_value_per_share"]

    expected_upside = (
        (intrinsic_value / 100.0) - 1
    ) * 100

    assert result["current_market_price"] == 100.0

    assert result["upside_percentage"] == pytest.approx(
        expected_upside
    )


# =====================================================
# 16. PROVIDER FAILURE
# =====================================================

def test_dcf_provider_failure(financial_agent_output):
    with patch(
        "tools.valuation_tools.get_statement",
        side_effect=ConnectionError(
            "Yahoo Finance unavailable"
        ),
    ):
        with pytest.raises(
            ConnectionError,
            match="Yahoo Finance unavailable",
        ):
            run_dcf_model(
                ticker="AAPL",
                financial_data=financial_agent_output,
            )


# =====================================================
# 17. FINANCIAL AGENT TICKER MISMATCH
# =====================================================

def test_dcf_financial_ticker_mismatch(
    financial_agent_output,
):
    financial_agent_output["ticker"] = "NVDA"

    with patch(
        "tools.valuation_tools.get_statement"
    ) as mock_statement:
        result = run_dcf_model(
            ticker="AAPL",
            financial_data=financial_agent_output,
        )

    assert result["data_status"] is False
    assert result["error"] == "Financial data ticker mismatch"

    mock_statement.assert_not_called()


# =====================================================
# 18. INCOMPLETE FINANCIAL AGENT DATA
# =====================================================

def test_dcf_incomplete_financial_agent_data(
    financial_agent_output,
):
    financial_agent_output["data_status"] = False

    with patch(
        "tools.valuation_tools.get_statement"
    ) as mock_statement:
        result = run_dcf_model(
            ticker="AAPL",
            financial_data=financial_agent_output,
        )

    assert result["data_status"] is False
    assert "insufficient data" in result["error"].lower()

    mock_statement.assert_not_called()


# =====================================================
# 19. VERIFY SHARED FINANCIAL DATA IS USED
# =====================================================

def test_dcf_uses_financial_agent_values(
    mock_yahoo,
    financial_agent_output,
):
    # Deliberately change values from the original
    # Yahoo mocks, proving the tool uses the agent output.
    financial_agent_output["operating_income"] = 120_000_000
    financial_agent_output["capital_expenditure"] = -20_000_000
    financial_agent_output["cash_and_equivalents"] = 30_000_000
    financial_agent_output["total_debt"] = 5_000_000

    result = run_dcf_model(
        ticker="AAPL",
        financial_data=financial_agent_output,
    )

    # 120M * 0.75 + 10M - 20M - 5M
    expected_fcff = 75_000_000

    assert result["data_status"] is True, result
    assert result["base_fcff"] == pytest.approx(expected_fcff)

    assert result["equity_value"] == pytest.approx(
        result["enterprise_value"]
        + 30_000_000
        - 5_000_000
    )
