def calculate_fcff(
        ebit:float,
        tax_rate:float,
        depreciation_amortization:float,
        capital_expenditure:float,
        working_capital_cash_flow:float
)->float:
    """
    Calculate FCFF.

    capital_expenditure is signed (negative).
    working_capital_cash_flow is the cash-flow statement's signed working-capital impact.
    """

    if not 0<=tax_rate<=1:
        raise ValueError("Tax rate must be between 0 and 1")

    return (
        ebit * (1 - tax_rate)
        + depreciation_amortization
        + capital_expenditure
        + working_capital_cash_flow
    )


def calculate_dcf(
        base_fcff: float,
        cash: float,
        total_debt: float,
        shares_outstanding: float,
        current_market_price: float,
        growth_rate: float = 0.06,
        discount_rate: float = 0.10,
        terminal_growth_rate: float = 0.025,
        forecast_years: int = 5
)->dict:
    """
    Five-year FCFF DCF using constant forecast growth and a perpetual-growth terminal value.
    """

    if base_fcff<=0:
        raise ValueError("Base FCFF must be positive")

    if shares_outstanding<=0:
        raise ValueError("Shares outstanding must be positive")

    if current_market_price<=0:
        raise ValueError("Market price must be positive")

    if not 1 <= forecast_years <= 20:
        raise ValueError("Invalid forecast period")

    if discount_rate <= terminal_growth_rate:
        raise ValueError(
            "Discount rate must exceed terminal growth rate"
        )

    if discount_rate <= 0 or growth_rate <= -1:
        raise ValueError("Invalid growth or discount rate")

    if cash < 0 or total_debt < 0:
        raise ValueError("Cash and debt cannot be negative")

    forecast=[]
    pv_cash_flows=0.0

    for year in range(1,forecast_years+1):
        fcff=base_fcff*(1+growth_rate)**year

        present_value=fcff/(1+discount_rate)**year

        pv_cash_flows+=present_value

        forecast.append({
            "year":year,
            "projected_fcff":fcff,
            "present_value":present_value
        })

    final_fcff=forecast[-1]["projected_fcff"]

    terminal_fcff=final_fcff*(1+terminal_growth_rate)

    terminal_value=terminal_fcff/(discount_rate-terminal_growth_rate)

    pv_terminal_value=terminal_value/(1+discount_rate)**forecast_years

    enterprise_value=pv_cash_flows+pv_terminal_value

    equity_value=enterprise_value+cash-total_debt

    intrinsic_value_per_share=equity_value/shares_outstanding

    upside_percentage=((intrinsic_value_per_share/current_market_price)-1)*100

    return {
        "base_fcff":base_fcff,
        "forecast": forecast,
        "pv_forecast_cash_flows": pv_cash_flows,
        "terminal_value": terminal_value,
        "pv_terminal_value": pv_terminal_value,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "intrinsic_value_per_share": intrinsic_value_per_share,
        "current_market_price": current_market_price,
        "upside_percentage": upside_percentage,
        "assumptions": {
            "growth_rate": growth_rate,
            "discount_rate": discount_rate,
            "terminal_growth_rate": terminal_growth_rate,
            "forecast_years": forecast_years
        }
    }