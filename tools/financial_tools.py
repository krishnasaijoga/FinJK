from strands import tool

from providers.yahoo_finance import get_metric, get_statement


@tool
def get_income_statement(ticker:str,period:str="annual")->dict:
    """
    Retrieve income statement of a company

    Args:
        ticker: ticker symbol for a company whose stock analysis query is searching. Eg:NVDA, APPL.
        period: annual or quarterly

    Returns:
        Dictionary containing company's income statement. Eg: {"ticker":"APPL","period":"annual","revenue":20000,"cost_of_revenue":100000,"operating_revenue":50000,"net_income":70000,"eps":4.7}
    """

    data=get_statement(ticker,"income",period)

    if not data["data_status"]:
        return data

    metrics=data["metrics"]

    revenue=get_metric(
        metrics,
        "Total Revenue",
        "TotalRevenue"
    )

    cost_of_revenue=get_metric(
        metrics,
        "Cost of Revenue",
        "Cost Of Revenue",
        "CostofRevenue",
        "CostOfRevenue"
    )

    operating_income=get_metric(
        metrics,
        "Operating Income",
        "OperatingIncome"
    )

    net_income=get_metric(
        metrics,
        "Net Income",
        "NetIncome"
    )

    eps=get_metric(
        metrics,
        "Diluted EPS",
        "DilutedEPS",
        "Basic EPS",
        "BasicEPS"
    )

    result={
        "ticker": data["ticker"],
        "period": data["period"],
        "report_date": data["report_date"],
        "revenue": revenue,
        "cost_of_revenue": cost_of_revenue,
        "operating_income": operating_income,
        "net_income": net_income,
        "eps": eps,
        "data_status": all(
            value is not None
            for value in (
                revenue,
                cost_of_revenue,
                operating_income,
                net_income,
            )
        ),
        "source": "Yahoo Finance",
    }

    return result


@tool
def get_balance_sheet(ticker:str,period:str="annual")->dict:
    """
    Retrieve a company's balance sheet
    
    Args:
        ticker: ticker symbol for a company whose stock analysis query is searching. Eg:NVDA, APPL.
        period: annual or quarterly

    Returns:
        Dictionary containing company's balance sheet. Eg: {"ticker":"APPL","total_assets":3500000,"total_liabilities":2800000,"shareholders_equity":7000000,"cash_and_equivalents":5000000,"total_debt":1000000}
    """

    data=get_statement(ticker,"balance",period)
    
    if not data["data_status"]:
        return data
    
    metrics=data["metrics"]

    total_assets=get_metric(
        metrics,
        "Total Assets",
        "TotalAssets"
    )

    total_liabilities=get_metric(
        metrics,
        "Total Liabilities Net Minority Interest",
        "TotalLiabilitiesNetMinorityInterest",
        "Total Liab"
    )

    shareholders_equity=get_metric(
        metrics,
        "Stockholders Equity",
        "StockholdersEquity",
        "Total Stockholder Equity",
        "Common Stock Equity",
        "CommonStockEquity"
    )

    cash_and_equivalents=get_metric(
        metrics,
        "Cash And Cash Equivalents",
        "CashAndCashEquivalents",
        "Cash"
    )

    total_debt=get_metric(
        metrics,
        "Total Debt",
        "TotalDebt"
    )

    result={
        "ticker": data["ticker"],
        "period": data["period"],
        "report_date": data["report_date"],
        "total_assets":total_assets,
        "total_liabilities":total_liabilities,
        "shareholders_equity":shareholders_equity,
        "cash_and_equivalents":cash_and_equivalents,
        "total_debt":total_debt,
        "data_status": all(
            value is not None
            for value in (
                total_assets,
                total_liabilities,
                shareholders_equity,
                cash_and_equivalents,
                total_debt
            )
        ),
        "source": "Yahoo Finance"
    }
    
    return result


@tool
def get_cash_flow_statement(ticker:str,period:str="annual")->dict:
    """
    Retrieves company's cash flow statement

    Args:
        ticker: ticker symbol for a company whose stock analysis query is searching. Eg:NVDA, APPL.
        period: annual or quarterly
    
    Returns:
        Dictionary containing company's cash flow statement. Eg: {"ticker":"APPL","operating_cash_flow":1100000,"investing_cash_flow": -2000000,"financing_cash_flow": -7000000,"capital_expenditure": 1500000,"free_cash_flow": 9500000}
    """

    data=get_statement(ticker,"cashflow",period)
        
    if not data["data_status"]:
        return data
    
    metrics=data["metrics"]

    operating_cash_flow=get_metric(
        metrics,
        "Operating Cash Flow",
        "OperatingCashFlow",
    )

    investing_cash_flow=get_metric(
        metrics,
        "Investing Cash Flow",
        "InvestingCashFlow",
    )

    financing_cash_flow=get_metric(
        metrics,
        "Financing Cash Flow",
        "FinancingCashFlow",
    )

    capital_expenditure=get_metric(
        metrics,
        "Capital Expenditure",
        "CapitalExpenditure",
    )

    free_cash_flow=get_metric(
        metrics,
        "Free Cash Flow",
        "FreeCashFlow",
    )

    if (free_cash_flow is None and operating_cash_flow is not None and capital_expenditure is not None):
        free_cash_flow=operating_cash_flow+capital_expenditure

    result={
        "ticker": data["ticker"],
        "period": data["period"],
        "report_date": data["report_date"],
        "operating_cash_flow":operating_cash_flow,
        "investing_cash_flow":investing_cash_flow,
        "financing_cash_flow":financing_cash_flow,
        "capital_expenditure":capital_expenditure,
        "free_cash_flow":free_cash_flow,
        "data_status": all(
            value is not None
            for value in (
                operating_cash_flow,
                investing_cash_flow,
                financing_cash_flow,
                capital_expenditure,
                free_cash_flow
            )
        ),
        "source": "Yahoo Finance"
    }

    return result


@tool
def compute_key_ratios(ticker:str)->dict:
    """
    Computes company's key ratios
    
    Args:
        ticker: ticker symbol for a company whose stock analysis query is searching. Eg:NVDA, APPL.
        
    Returns:
        Dictionary containing company's computed key ratios. Eg: {"ticker":"APPL","ratios":{"net_profit_margin":0.2346,"return_on_equity":1.3751,"debt_to_equity":1.4267,"free_cash_flow_margin":0.2345}}
    """
    ticker=ticker.upper()

    # later add cache
    income=get_income_statement(ticker)
    balance=get_balance_sheet(ticker)
    cash_flow=get_cash_flow_statement(ticker)

    revenue=income["revenue"]
    net_income=income["net_income"]
    shareholders_equity=balance["shareholders_equity"]
    total_debt=balance["total_debt"]

    free_cash_flow=cash_flow["free_cash_flow"]

    net_profit_margin=net_income/revenue if revenue!=0 else None

    debt_to_equity=total_debt/shareholders_equity if shareholders_equity!=0 else None

    return_on_equity=net_income/shareholders_equity if shareholders_equity!=0 else None

    free_cash_flow_margin=free_cash_flow/revenue if revenue!=0 else None

    mock_key_ratios={
        "ticker":ticker,
        "ratios":{
            "net_profit_margin":net_profit_margin,  #later add _float_clean()
            "return_on_equity":return_on_equity,
            "debt_to_equity":debt_to_equity,
            "free_cash_flow_margin":free_cash_flow_margin
        }
    }

    return mock_key_ratios