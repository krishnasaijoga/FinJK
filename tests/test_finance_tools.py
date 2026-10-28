import pytest

from tools.financial_tools import (
    compute_key_ratios,
    get_balance_sheet,
    get_cash_flow_statement,
    get_income_statement,
)


@pytest.fixture
def ticker():
    return "AAPL"


# TEST 1: Income Statement
def test_get_income_statement(ticker):
    result = get_income_statement(ticker)

    assert isinstance(result, dict)
    assert result["ticker"] == ticker
    assert result["period"] == "annual"

    assert result["revenue"] > 0
    assert result["cost_of_revenue"] >= 0
    assert result["operating_income"] <= (
        result["revenue"] - result["cost_of_revenue"]
    )
    assert result["eps"] > 0


# TEST 2: Balance Sheet
def test_get_balance_sheet(ticker):
    result = get_balance_sheet(ticker)

    assert isinstance(result, dict)
    assert result["ticker"] == ticker

    assert result["total_assets"] == (
        result["total_liabilities"]
        + result["shareholders_equity"]
    )

    assert result["total_debt"] >= 0
    assert result["cash_and_equivalents"] >= 0


# TEST 3: Cash Flow Statement
def test_get_cash_flow_statement(ticker):
    result = get_cash_flow_statement(ticker)

    assert isinstance(result, dict)
    assert result["ticker"] == ticker

    expected_fcf = (
        result["operating_cash_flow"]
        - result["capital_expenditure"]
    )

    assert result["free_cash_flow"] == expected_fcf


# TEST 4: Key Ratios
def test_compute_key_ratios(ticker):
    result = compute_key_ratios(ticker)

    income = get_income_statement(ticker)
    balance = get_balance_sheet(ticker)
    cash_flow = get_cash_flow_statement(ticker)

    ratios = result["ratios"]

    assert result["ticker"] == ticker

    assert ratios["net_profit_margin"] == pytest.approx(
        income["net_income"] / income["revenue"],
        abs=0.0001
    )

    assert ratios["return_on_equity"] == pytest.approx(
        income["net_income"] / balance["shareholders_equity"],
        abs=0.0001
    )

    assert ratios["debt_to_equity"] == pytest.approx(
        balance["total_debt"] / balance["shareholders_equity"],
        abs=0.0001
    )

    assert ratios["free_cash_flow_margin"] == pytest.approx(
        cash_flow["free_cash_flow"] / income["revenue"],
        abs=0.0001
    )


# TEST 5: Ticker Normalization
@pytest.mark.parametrize(
    "tool_function",
    [
        get_income_statement,
        get_balance_sheet,
        get_cash_flow_statement,
        compute_key_ratios
    ]
)
def test_ticker_normalization(tool_function):
    result = tool_function("aapl")

    assert result["ticker"] == "AAPL"
