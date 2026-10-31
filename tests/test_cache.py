
from unittest.mock import patch

from tools.financial_tools import _get_balance_sheet_cached


from unittest.mock import patch

from tools.financial_tools import _get_balance_sheet_cached


from unittest.mock import patch

from tools.financial_tools import (
    compute_key_ratios,
    _compute_key_ratios_cached,
)


from unittest.mock import patch

from tools.news_tools import (
    search_recent_news,
    _search_recent_news_cached,
)


from cachetools import TTLCache, cached


import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from tools.financial_tools import (
    _compute_key_ratios_cached,
)


import pytest
from pydantic import ValidationError
from strands.multiagent.base import Status

from agents.orchestrator import get_validated_output
from agents.financial_data_agent import FinancialDataOutput


def test_parallel_ratio_cache_requests():
    _compute_key_ratios_cached.cache_clear()

    started = threading.Barrier(2)
    calls = 0
    calls_lock = threading.Lock()

    def mock_income(ticker):
        nonlocal calls

        with calls_lock:
            calls += 1

        # Give both callers an opportunity
        # to encounter the same cache miss.
        time.sleep(0.1)

        return {
            "ticker": ticker,
            "revenue": 100_000_000,
            "net_income": 20_000_000,
        }

    def mock_balance(ticker):
        return {
            "ticker": ticker,
            "shareholders_equity": 50_000_000,
            "total_debt": 10_000_000,
        }

    def mock_cashflow(ticker):
        return {
            "ticker": ticker,
            "free_cash_flow": 15_000_000,
        }

    def worker():
        started.wait()
        return _compute_key_ratios_cached("AAPL")

    with (
        patch(
            "tools.financial_tools.get_income_statement",
            side_effect=mock_income,
        ),
        patch(
            "tools.financial_tools.get_balance_sheet",
            side_effect=mock_balance,
        ),
        patch(
            "tools.financial_tools.get_cash_flow_statement",
            side_effect=mock_cashflow,
        ),
    ):
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(worker)
                for _ in range(2)
            ]

            results = [
                future.result()
                for future in futures
            ]

    assert results[0] == results[1]

    print(
        f"\nUnderlying income calls: {calls}"
    )

    # Both requests must return correct results.
    # Duplicate concurrent computations are
    # permitted by the current cache design.
    assert calls in (1, 2)


def test_cache_ttl_expiration():
    clock = [0.0]
    cache = TTLCache(
        maxsize=10,
        ttl=10,
        timer=lambda: clock[0],
    )

    calls = []

    @cached(cache=cache)
    def fetch_data(ticker):
        calls.append(ticker)
        return {"ticker": ticker, "version": len(calls)}

    first = fetch_data("AAPL")
    second = fetch_data("AAPL")

    assert first == second
    assert len(calls) == 1

    # Advance simulated time beyond the TTL.
    clock[0] = 11.0

    third = fetch_data("AAPL")

    assert len(calls) == 2
    assert third["version"] == 2


def mock_news_articles(ticker):
    return [
        {
            "ticker": ticker,
            "title": f"{ticker} reports strong earnings",
            "description": "The company reported revenue growth.",
            "published_at": "2026-10-29T10:00:00Z",
            "source": "Test News",
            "url": "https://example.com/news1",
        },
        {
            "ticker": ticker,
            "title": f"{ticker} announces new investment",
            "description": "The company announced expansion plans.",
            "published_at": "2026-10-29T12:00:00Z",
            "source": "Test News",
            "url": "https://example.com/news2",
        },
    ]


# TEST 1: Identical searches reuse cached results
def test_news_cache_hit():
    _search_recent_news_cached.cache_clear()

    with patch(
        "tools.news_tools.fetch_company_news",
        side_effect=lambda ticker, company_name, days:
            mock_news_articles(ticker),
    ) as mock_provider:

        first = search_recent_news("AAPL", days=5)
        second = search_recent_news("AAPL", days=5)

    assert first == second
    assert first["articles_found"] == 2
    assert first["data_status"] is True
    assert mock_provider.call_count == 1


# TEST 2: Different tickers use separate cache entries
def test_news_cache_different_tickers():
    _search_recent_news_cached.cache_clear()

    with patch(
        "tools.news_tools.fetch_company_news",
        side_effect=lambda ticker, company_name, days:
            mock_news_articles(ticker),
    ) as mock_provider:

        aapl = search_recent_news("AAPL", days=5)
        msft = search_recent_news("MSFT", days=5)
        aapl_again = search_recent_news("AAPL", days=5)

    assert aapl["ticker"] == "AAPL"
    assert msft["ticker"] == "MSFT"
    assert aapl_again == aapl
    assert mock_provider.call_count == 2


# TEST 3: Different lookback periods use separate cache entries
def test_news_cache_different_days():
    _search_recent_news_cached.cache_clear()

    with patch(
        "tools.news_tools.fetch_company_news",
        side_effect=lambda ticker, company_name, days:
            mock_news_articles(ticker),
    ) as mock_provider:

        search_recent_news("AAPL", days=5)
        search_recent_news("AAPL", days=30)
        search_recent_news("AAPL", days=5)

    assert mock_provider.call_count == 2


# TEST 4: Company-name normalization
def test_news_cache_company_name_normalization():
    _search_recent_news_cached.cache_clear()

    with patch(
        "tools.news_tools.fetch_company_news",
        side_effect=lambda ticker, company_name, days:
            mock_news_articles(ticker),
    ) as mock_provider:

        first = search_recent_news(
            " aapl ",
            company_name=" Apple Inc ",
            days=5,
        )

        second = search_recent_news(
            "AAPL",
            company_name="apple inc",
            days=5,
        )

    assert first == second
    assert mock_provider.call_count == 1


# TEST 5: Different company names use different entries
def test_news_cache_different_company_names():
    _search_recent_news_cached.cache_clear()

    with patch(
        "tools.news_tools.fetch_company_news",
        side_effect=lambda ticker, company_name, days:
            mock_news_articles(ticker),
    ) as mock_provider:

        search_recent_news(
            "AAPL",
            company_name="Apple Inc",
        )

        search_recent_news(
            "AAPL",
            company_name="Apple",
        )

    assert mock_provider.call_count == 2


# TEST 6: Protect cached results from mutation
def test_news_cache_deepcopy():
    _search_recent_news_cached.cache_clear()

    with patch(
        "tools.news_tools.fetch_company_news",
        side_effect=lambda ticker, company_name, days:
            mock_news_articles(ticker),
    ) as mock_provider:

        first = search_recent_news("AAPL", days=5)

        first["news"][0]["title"] = "MODIFIED TITLE"

        second = search_recent_news("AAPL", days=5)

    assert second["news"][0]["title"] == (
        "AAPL reports strong earnings"
    )

    assert mock_provider.call_count == 1


def mock_income(ticker):
    return {
        "ticker": ticker,
        "revenue": 100_000_000,
        "net_income": 20_000_000,
    }


def mock_balance(ticker):
    return {
        "ticker": ticker,
        "shareholders_equity": 50_000_000,
        "total_debt": 10_000_000,
    }


def mock_cashflow(ticker):
    return {
        "ticker": ticker,
        "free_cash_flow": 15_000_000,
    }


# TEST 1: Identical requests use cached results
def test_ratios_cache_hit():
    _compute_key_ratios_cached.cache_clear()

    with (
        patch(
            "tools.financial_tools.get_income_statement",
            side_effect=mock_income,
        ) as income,
        patch(
            "tools.financial_tools.get_balance_sheet",
            side_effect=mock_balance,
        ) as balance,
        patch(
            "tools.financial_tools.get_cash_flow_statement",
            side_effect=mock_cashflow,
        ) as cashflow,
    ):
        first = compute_key_ratios("AAPL")
        second = compute_key_ratios("AAPL")

    assert first == second

    assert income.call_count == 1
    assert balance.call_count == 1
    assert cashflow.call_count == 1

    assert first["ratios"]["net_profit_margin"] == 0.20
    assert first["ratios"]["return_on_equity"] == 0.40
    assert first["ratios"]["debt_to_equity"] == 0.20
    assert first["ratios"]["free_cash_flow_margin"] == 0.15


# TEST 2: Different tickers have separate cache entries
def test_ratios_cache_different_tickers():
    _compute_key_ratios_cached.cache_clear()

    with (
        patch(
            "tools.financial_tools.get_income_statement",
            side_effect=mock_income,
        ) as income,
        patch(
            "tools.financial_tools.get_balance_sheet",
            side_effect=mock_balance,
        ) as balance,
        patch(
            "tools.financial_tools.get_cash_flow_statement",
            side_effect=mock_cashflow,
        ) as cashflow,
    ):
        aapl = compute_key_ratios("AAPL")
        msft = compute_key_ratios("MSFT")
        aapl_again = compute_key_ratios("AAPL")

    assert aapl["ticker"] == "AAPL"
    assert msft["ticker"] == "MSFT"
    assert aapl_again == aapl

    assert income.call_count == 2
    assert balance.call_count == 2
    assert cashflow.call_count == 2


# TEST 3: Ticker normalization
def test_ratios_cache_ticker_normalization():
    _compute_key_ratios_cached.cache_clear()

    with (
        patch(
            "tools.financial_tools.get_income_statement",
            side_effect=mock_income,
        ) as income,
        patch(
            "tools.financial_tools.get_balance_sheet",
            side_effect=mock_balance,
        ) as balance,
        patch(
            "tools.financial_tools.get_cash_flow_statement",
            side_effect=mock_cashflow,
        ) as cashflow,
    ):
        first = compute_key_ratios(" aapl ")
        second = compute_key_ratios("AAPL")

    assert first == second
    assert income.call_count == 1
    assert balance.call_count == 1
    assert cashflow.call_count == 1


# TEST 4: Cached results are protected from mutation
def test_ratios_cache_deepcopy():
    _compute_key_ratios_cached.cache_clear()

    with (
        patch(
            "tools.financial_tools.get_income_statement",
            side_effect=mock_income,
        ),
        patch(
            "tools.financial_tools.get_balance_sheet",
            side_effect=mock_balance,
        ),
        patch(
            "tools.financial_tools.get_cash_flow_statement",
            side_effect=mock_cashflow,
        ),
    ):
        first = compute_key_ratios("AAPL")

        first["ratios"]["net_profit_margin"] = 999

        second = compute_key_ratios("AAPL")

    assert second["ratios"]["net_profit_margin"] == 0.20


def make_balance_statement(ticker="AAPL", period="annual"):
    return {
        "ticker": ticker,
        "period": period,
        "data_status": True,
        "report_date": "2025-09-30",
        "metrics": {
            "Total Assets": 100_000_000,
            "Total Liabilities Net Minority Interest": 60_000_000,
            "Stockholders Equity": 40_000_000,
            "Cash And Cash Equivalents": 10_000_000,
            "Total Debt": 20_000_000,
        },
    }


def test_cache_different_tickers():
    _get_balance_sheet_cached.cache_clear()

    with patch(
        "tools.financial_tools.get_statement",
        side_effect=lambda ticker, statement, period:
            make_balance_statement(ticker, period),
    ) as mock_provider:

        _get_balance_sheet_cached("AAPL", "annual")
        _get_balance_sheet_cached("MSFT", "annual")
        _get_balance_sheet_cached("AAPL", "annual")

    assert mock_provider.call_count == 2


def test_cache_different_periods():
    _get_balance_sheet_cached.cache_clear()

    with patch(
        "tools.financial_tools.get_statement",
        side_effect=lambda ticker, statement, period:
            make_balance_statement(ticker, period),
    ) as mock_provider:

        _get_balance_sheet_cached("AAPL", "annual")
        _get_balance_sheet_cached("AAPL", "quarterly")
        _get_balance_sheet_cached("AAPL", "annual")

    assert mock_provider.call_count == 2


def test_cache_returns_same_values():
    _get_balance_sheet_cached.cache_clear()

    with patch(
        "tools.financial_tools.get_statement",
        return_value=make_balance_statement(),
    ) as mock_provider:

        first = _get_balance_sheet_cached("AAPL", "annual")
        second = _get_balance_sheet_cached("AAPL", "annual")

    assert first == second
    assert first["data_status"] is True
    assert mock_provider.call_count == 1


def test_balance_sheet_cache():
    _get_balance_sheet_cached.cache_clear()

    fake_statement = {
        "ticker": "AAPL",
        "period": "annual",
        "data_status": True,
        "report_date": "2025-09-30",
        "metrics": {
            "Total Assets": 100_000_000,
            "Total Liabilities Net Minority Interest": 60_000_000,
            "Stockholders Equity": 40_000_000,
            "Cash And Cash Equivalents": 10_000_000,
            "Total Debt": 20_000_000,
        },
    }

    with patch(
        "tools.financial_tools.get_statement",
        return_value=fake_statement,
    ) as mock_provider:

        first = _get_balance_sheet_cached(
            "AAPL", "annual"
        )

        second = _get_balance_sheet_cached(
            "AAPL", "annual"
        )
    

    assert first == second
    assert mock_provider.call_count == 1
