
import pytest

from tools.valuation_tools import get_peer_multiples, run_dcf_model


@pytest.fixture
def ticker():
    return "AAPL"


# TEST 1: DCF Model
def test_run_dcf_model(ticker):
    result = run_dcf_model(ticker)

    assert isinstance(result, dict)
    assert result["ticker"] == ticker
    assert result["valuation_method"] == "DCF"

    assert result["enterprise_value"] > 0
    assert result["equity_value"] > 0
    assert result["intrinsic_value_per_share"] > 0
    assert result["current_market_price"] > 0

    # Verify upside percentage calculation
    expected_upside = (
        (result["intrinsic_value_per_share"]
         - result["current_market_price"])
        / result["current_market_price"]
    ) * 100

    assert result["upside_percentage"] == pytest.approx(
        expected_upside, abs=0.01
    )

    assert result["valuation_status"] in [
        "Undervalued", "Overvalued", "Fairly Valued"
    ]


# TEST 2: Peer Multiples
def test_get_peer_multiples(ticker):
    result = get_peer_multiples(ticker)

    assert isinstance(result, dict)
    assert result["ticker"] == ticker
    assert result["valuation_method"] == "Peer Multiples"

    # Verify peer tickers
    assert isinstance(result["peer_tickers"], list)
    assert len(result["peer_tickers"]) > 0

    # Verify company valuation metrics
    for metric in ["pe_ratio", "ev_ebitda", "ps_ratio"]:
        assert metric in result["company_multiples"]
        assert metric in result["peer_average_multiples"]

        assert isinstance(
            result["company_multiples"][metric], (int, float)
        )

        assert isinstance(
            result["peer_average_multiples"][metric], (int, float)
        )

    assert result["relative_valuation"] == "Trading at a discount"
    assert isinstance(result["summary"], str)


# TEST 3: Ticker Normalization
@pytest.mark.parametrize(
    "tool_function",
    [run_dcf_model, get_peer_multiples]
)
def test_ticker_normalization(tool_function):
    result = tool_function("aapl")

    assert result["ticker"] == "AAPL"