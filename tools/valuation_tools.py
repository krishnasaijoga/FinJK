from strands import tool

from statistics import mean
import math

from providers.yahoo_finance import get_metric, get_statement, get_stock

from tools.dcf_calculator import calculate_dcf, calculate_fcff

@tool
def run_dcf_model(
    ticker:str,
    financial_data:dict,
    growth_rate: float = 0.06,
    discount_rate: float = 0.10,
    terminal_growth_rate: float = 0.025,
    forecast_years: int = 5
)->dict:
    """
    Estimates Discounted Cash Flow worth of a company today based on its other financial metrics

    Args:
        ticker: ticker symbol for a company whose stock analysis query is searching. Eg:NVDA, APPL.
        growth_rate: Annual projected FCFF growth.
        discount_rate: Discount rate (WACC assumption).
        terminal_growth_rate: Long-term terminal growth.
        forecast_years: Explicit forecast period.
    
    Returns:
        Dictionary containing company's dcf related metrics. Eg: {"ticker":"APPL","valuation_method": "DCF","enterprise_value": 1800000000000,"equity_value": 1700000000000,"intrinsic_value_per_share": 113.33,"current_market_price": 100.00,"upside_percentage": 13.33,"valuation_status": "Undervalued"}
    """

    ticker=ticker.strip().upper()

    if not ticker:
        raise ValueError("Ticker cannot be empty")

    if not isinstance(financial_data,dict):
        raise ValueError("financial_data must be a dictionary")

    if financial_data.get("ticker", "").upper() != ticker:
        return {
            "ticker": ticker,
            "data_status": False,
            "error": "Financial data ticker mismatch",
        }

    if not financial_data.get("data_status", False):
        return {
            "ticker": ticker,
            "data_status": False,
            "error": "Financial agent returned insufficient data",
        }

    period="annual"
    income=get_statement(ticker,"income",period)
    cashflow=get_statement(ticker,"cashflow",period)
    balance=get_statement(ticker,"balance",period)

    if not all(x["data_status"] for x in (income,cashflow,balance)):
        return {
            "ticker": ticker,
            "data_status": False,
            "error": "Required financial statements unavailable",
        }

    income_metrics=income["metrics"]
    cashflow_metrics=cashflow["metrics"]
    balance_metrics=balance["metrics"]

    ebit = financial_data.get("operating_income")

    pretax_income = get_metric(
        income_metrics,
        "Pretax Income",
        "PretaxIncome",
    )

    tax_provision = get_metric(
        income_metrics,
        "Tax Provision",
        "TaxProvision",
    )

    # Cash flow statement
    depreciation = get_metric(
        cashflow_metrics,
        "Depreciation And Amortization",
        "DepreciationAndAmortization",
        "Depreciation Amortization Depletion",
        "DepreciationAmortizationDepletion",
    )

    capex = financial_data.get("capital_expenditure")

    working_capital_cash_flow = get_metric(
        cashflow_metrics,
        "Change In Working Capital",
        "ChangeInWorkingCapital",
    )

    # Balance sheet
    cash = financial_data.get("cash_and_equivalents")

    total_debt = financial_data.get("total_debt")

    stock=get_stock(ticker)
    info=stock.info

    shares_outstanding=info.get("sharesOutstanding")

    current_price=(
        info.get("currentPrice") or info.get("regularMarketPrice")
    )

    required={
        "ebit":ebit,
        "pretax_income": pretax_income,
        "tax_provision": tax_provision,
        "depreciation": depreciation,
        "capex": capex,
        "working_capital_cash_flow": working_capital_cash_flow,
        "cash": cash,
        "total_debt": total_debt,
        "shares_outstanding": shares_outstanding,
        "current_price": current_price
    }

    missing=[
        name for name, value in required.items() if value is None
    ]

    if missing:
        return {
            "ticker":ticker,
            "data_status":False,
            "missing_fields":missing,
            "error":"insifficient inputs for FCFF DCF"
        }

    if pretax_income is None or pretax_income <=0:
        return {
            "ticker": ticker,
            "data_status": False,
            "error": "Cannot derive a valid tax rate",
        }

    if (
        ebit is None
        or pretax_income is None
        or tax_provision is None
        or depreciation is None
        or capex is None
        or working_capital_cash_flow is None
        or cash is None
        or total_debt is None
        or shares_outstanding is None
        or current_price is None
    ):
        return {
            "ticker": ticker,
            "data_status": False,
            "missing_fields": missing,
            "error": "Insufficient inputs for FCFF DCF",
        }

    tax_rate=tax_provision/pretax_income

    if not 0 <= tax_rate <= 1:
        return {
            "ticker": ticker,
            "data_status": False,
            "error": "Invalid effective tax rate",
        }
    
    try:
        base_fcff=calculate_fcff(
            ebit=ebit,
            tax_rate=tax_rate,
            depreciation_amortization=depreciation,
            capital_expenditure=capex,
            working_capital_cash_flow=working_capital_cash_flow
        )

        dcf = calculate_dcf(
            base_fcff=base_fcff,
            cash=cash,
            total_debt=total_debt,
            shares_outstanding=shares_outstanding,
            current_market_price=current_price,
            growth_rate=growth_rate,
            discount_rate=discount_rate,
            terminal_growth_rate=terminal_growth_rate,
            forecast_years=forecast_years,
        )
    except ValueError as exc:
        return {
            "ticker":ticker,
            "data_status":False,
            "error":str(exc)
        }

    upside=dcf["upside_percentage"]

    if upside>10:
        valuation_status="Undervalued"
    elif upside<-10:
        valuation_status="Overvalued"
    else:
        valuation_status="Fairly valued"

    dcf_results={
        "ticker": ticker,
        "data_status": True,
        "source": "Yahoo Finance",
        "financial_input_source": "FinancialDataAgent",
        "valuation_method": "DCF",
        "model": "Five-year FCFF DCF",
        "valuation_status":valuation_status,
        "report_dates": {
            "income": income["report_date"],
            "cashflow": cashflow["report_date"],
            "balance": balance["report_date"],
        },
        "tax_rate": tax_rate,
        **dcf,
    }
    
    return dcf_results


def valid_multiple(value):
    """Return a positive finite multiple, or None"""
    try:
        number=float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number>0 else None


def get_company_multiples(ticker:str)->dict:
    """Fetch one company's valuation multiples."""
    info=get_stock(ticker).info

    return {
        "ticker": ticker,
        "company_name": info.get("longName", ticker),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "currency": info.get("currency"),
        "pe_ratio": valid_multiple(info.get("trailingPE")),
        "ev_ebitda": valid_multiple(
            info.get("enterpriseToEbitda")
        ),
        "ps_ratio": valid_multiple(
            info.get("priceToSalesTrailing12Months")
        )
    }


@tool
def get_peer_multiples(ticker:str,peer_tickers:list[str]|None=None)->dict:
    """
    Compare a company's valuation multipes with peers.

    Args:
        ticker: ticker symbol for a company whose stock analysis query is searching. Eg:NVDA, APPL.
        peer_tickers: Selected comparable company tickers.
            Do not include the target ticker.
    
    Returns:
        Company multiples, peer averages, comparisons
        and relative valuation assessment.
    """

    ticker=ticker.strip().upper()

    if not ticker:
        raise ValueError("Ticker cannot be empty")

    peers=list(dict.fromkeys(
        p.strip().upper()
        for p in (peer_tickers or [])
        if p.strip()
    ))

    peers=[p for p in peers if p!=ticker]

    if not peers:
        return {
            "ticker": ticker,
            "valuation_method": "Peer Multiples",
            "peer_tickers": [],
            "data_status": False,
            "error": "No peer companies supplied"
        }

    company=get_company_multiples(ticker)

    peer_data=[]
    failed_peers=[]

    for peer in peers:
        try:
            data=get_company_multiples(peer)
            if any(
                data[key] is not None
                for key in ("pe_ratio","ev_ebitda","ps_ratio")
            ):
                peer_data.append(data)
            else:
                failed_peers.append(data)
        except Exception:
            failed_peers.append(peer)

    multiple_keys=("pe_ratio","ev_ebitda","ps_ratio")

    peer_averages={}
    peer_counts={}

    for key in multiple_keys:
        values=[
            peer[key] for peer in peer_data if peer[key] is not None
        ]
        peer_counts[key]=len(values)
        peer_averages[key]=mean(values) if values else None

    comparisons={}

    for key in multiple_keys:
        own_value=company[key]
        peer_average=peer_averages[key]

        if own_value is None or peer_average is None:
            comparisons[key]=None
        else:
            comparisons[key]=((own_value/peer_average)-1)*100

    available=[
        value
        for value in comparisons.values()
        if value is not None
    ]

    if len(available) < 2:
        relative_valuation = "INSUFFICIENT_DATA"
    elif all(value < 0 for value in available):
        relative_valuation = "DISCOUNT"
    elif all(value > 0 for value in available):
        relative_valuation = "PREMIUM"
    else:
        relative_valuation = "MIXED"

    data_status=(len(peer_data)>0 and len(available)>=2)

    summary={
        f"{ticker} relative valuation: "
        f"{relative_valuation}. "
        f"Compared with {len(peer_data)} peers. "
        "Negative comparison percentages indicate "
        "multiples below peer averages."
    }

    result={
        "ticker": ticker,
        "valuation_method": "Peer Multiples",
        "peer_tickers": [
            peer["ticker"] for peer in peer_data
        ],
        "company_multiples": {
            key: company[key] for key in multiple_keys
        },
        "peer_average_multiples": peer_averages,
        "peer_data": peer_data,
        "peer_counts": peer_counts,
        "comparison_percentages": comparisons,
        "failed_peers": failed_peers,
        "relative_valuation": relative_valuation,
        "summary": summary,
        "source": "Yahoo Finance",
        "data_status": data_status,
    }

    return result