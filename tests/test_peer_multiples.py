
from unittest.mock import patch

import pytest

from tools.valuation_tools import (
    get_peer_multiples,
    valid_multiple,
)


@pytest.fixture
def mock_peer_info():
    return {
        "AAPL": {
            "longName": "Target Company",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "currency": "USD",
            "trailingPE": 24.0,
            "enterpriseToEbitda": 12.0,
            "priceToSalesTrailing12Months": 4.0,
        },
        "PEER1": {
            "longName": "Peer One",
            "sector": "Technology",
            "currency": "USD",
            "trailingPE": 30.0,
            "enterpriseToEbitda": 15.0,
            "priceToSalesTrailing12Months": 5.0,
        },
        "PEER2": {
            "longName": "Peer Two",
            "sector": "Technology",
            "currency": "USD",
            "trailingPE": 30.0,
            "enterpriseToEbitda": 15.0,
            "priceToSalesTrailing12Months": 5.0,
        },
    }


@pytest.fixture
def mock_yahoo(mock_peer_info):
    def fake_get_stock(ticker):
        if ticker not in mock_peer_info:
            raise ValueError("Unknown ticker")

        class Stock:
            info = mock_peer_info[ticker]

        return Stock()

    with patch(
        "tools.valuation_tools.get_stock",
        side_effect=fake_get_stock,
    ):
        yield


def test_valid_multiple():
    assert valid_multiple(24) == 24
    assert valid_multiple(None) is None
    assert valid_multiple(-5) is None
    assert valid_multiple(float("nan")) is None


def test_peer_multiples_success(mock_yahoo):
    result = get_peer_multiples(
        "AAPL", ["PEER1", "PEER2"]
    )

    assert result["data_status"] is True
    assert result["peer_average_multiples"]["pe_ratio"] == 30
    assert result["peer_average_multiples"]["ev_ebitda"] == 15
    assert result["peer_average_multiples"]["ps_ratio"] == 5
    assert result["relative_valuation"] == "DISCOUNT"


def test_comparison_percentage(mock_yahoo):
    result = get_peer_multiples(
        "AAPL", ["PEER1", "PEER2"]
    )

    assert result["comparison_percentages"][
        "pe_ratio"
    ] == pytest.approx(-20)


def test_missing_peers():
    result = get_peer_multiples("AAPL")

    assert result["data_status"] is False
    assert result["peer_tickers"] == []


def test_duplicate_peers_removed(mock_yahoo):
    result = get_peer_multiples(
        "AAPL",
        ["PEER1", "PEER1", "AAPL", "PEER2"],
    )

    assert result["peer_tickers"] == ["PEER1", "PEER2"]


def test_failed_peer_is_reported(mock_yahoo):
    result = get_peer_multiples(
        "AAPL", ["PEER1", "BAD"]
    )

    assert "BAD" in result["failed_peers"]
    assert result["data_status"] is True


def test_invalid_target_ticker():
    with pytest.raises(ValueError):
        get_peer_multiples("  ")


def test_partial_peer_metric(mock_yahoo, mock_peer_info):
    mock_peer_info["PEER2"]["trailingPE"] = None

    result = get_peer_multiples(
        "AAPL", ["PEER1", "PEER2"]
    )

    assert result["peer_counts"]["pe_ratio"] == 1
    assert result["peer_average_multiples"]["pe_ratio"] == 30


def test_mixed_valuation(mock_yahoo, mock_peer_info):
    mock_peer_info["AAPL"]["trailingPE"] = 35

    result = get_peer_multiples(
        "AAPL", ["PEER1", "PEER2"]
    )

    assert result["relative_valuation"] == "MIXED"
