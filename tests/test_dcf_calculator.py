
import pytest

from tools.dcf_calculator import (
    calculate_fcff,
    calculate_dcf,
)


def test_fcff_calculation():
    result = calculate_fcff(
        ebit=100_000_000,
        tax_rate=0.25,
        depreciation_amortization=10_000_000,
        capital_expenditure=-15_000_000,
        working_capital_cash_flow=-5_000_000,
    )

    assert result == 65_000_000


def test_dcf_returns_expected_fields():
    result = calculate_dcf(
        base_fcff=65_000_000,
        cash=10_000_000,
        total_debt=20_000_000,
        shares_outstanding=10_000_000,
        current_market_price=100,
    )

    assert len(result["forecast"]) == 5
    assert result["enterprise_value"] > 0
    assert result["equity_value"] == pytest.approx(
        result["enterprise_value"] - 10_000_000
    )
    assert result["intrinsic_value_per_share"] == pytest.approx(
        result["equity_value"] / 10_000_000
    )
    assert result["upside_percentage"] == pytest.approx(
        (result["intrinsic_value_per_share"] / 100 - 1) * 100
    )


def test_invalid_discount_rate():
    with pytest.raises(ValueError):
        calculate_dcf(
            base_fcff=65_000_000,
            cash=10_000_000,
            total_debt=20_000_000,
            shares_outstanding=10_000_000,
            current_market_price=100,
            discount_rate=0.02,
            terminal_growth_rate=0.025,
        )


def test_invalid_share_count():
    with pytest.raises(ValueError):
        calculate_dcf(
            base_fcff=65_000_000,
            cash=10_000_000,
            total_debt=20_000_000,
            shares_outstanding=0,
            current_market_price=100,
        )


def test_fcff_growth():
    result = calculate_dcf(
        base_fcff=100_000_000,
        cash=0,
        total_debt=0,
        shares_outstanding=10_000_000,
        current_market_price=100,
        growth_rate=0.06,
    )

    assert result["forecast"][0]["projected_fcff"] == pytest.approx(
        106_000_000
    )

    assert result["forecast"][1]["projected_fcff"] == pytest.approx(
        112_360_000
    )
