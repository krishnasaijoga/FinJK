
import pytest

from tools.valuation_tools import get_peer_multiples, run_dcf_model


@pytest.fixture
def ticker():
    return "AAPL"

@pytest.fixture
def mock_financial_output():
    return {
        "ticker": "AAPL",
        "data_status": True,
        "operating_income": 100_000_000,
        "cash_and_equivalents": 10_000_000,
        "total_debt": 20_000_000,
        "capital_expenditure": -15_000_000,
    }


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


# TEST 1: DCF Model
def test_run_dcf_model(mock_yahoo, financial_agent_output):
    result = run_dcf_model(
        ticker="AAPL",
        financial_data=financial_agent_output
    )

    assert result["data_status"] is True, result
    assert result["ticker"] == "AAPL"
    assert result["valuation_method"] == "DCF"
    assert result["enterprise_value"] > 0


# TEST 2: Peer Multiples
def test_get_peer_multiples_missing_peers(ticker):
    result = get_peer_multiples(ticker)

    assert isinstance(result, dict)
    assert result["ticker"] == ticker
    assert result["valuation_method"] == "Peer Multiples"
    assert result["peer_tickers"] == []
    assert result["data_status"] is False
    assert result["error"] == "No peer companies supplied"


# TEST 3: Ticker Normalization
@pytest.mark.parametrize(
    "tool_func",
    [run_dcf_model, get_peer_multiples]
)
def test_ticker_normalization(
    tool_func,
    mock_yahoo,
    financial_agent_output
):
    if tool_func == run_dcf_model:
        result = tool_func(
            ticker="aapl",
            financial_data=financial_agent_output
        )
    else:
        result = tool_func(
            ticker="aapl",
            peer_tickers=[]
        )

    assert result["ticker"] == "AAPL"