from strands import tool


@tool
def get_income_statement(ticker:str)->dict:
    """
    Retrieve income statement of a company

    Args:
        ticker: ticker symbol for a company whose stock analysis query is searching. Eg:NVDA, APPL.

    Returns:
        Dictionary containing company's income statement. Eg: {"ticker":"APPL","period":"annual","revenue":20000,"cost_of_revenue":100000,"operating_revenue":50000,"net_income":70000,"eps":4.7}
    """

    ticker=ticker.upper()
    mock_income_statement={
        "ticker":ticker.upper(),
        "period":"annual",
        "revenue":4000000,
        "cost_of_revenue":2200000,
        "operating_income":1800000,
        "net_income":9500000,
        "eps":6.25
    }

    return mock_income_statement


@tool
def get_balance_sheet(ticker:str)->dict:
    """
    Retrieve a company's balance sheet
    
    Args:
        ticker: ticker symbol for a company whose stock analysis query is searching. Eg:NVDA, APPL.

    Returns:
        Dictionary containing company's balance sheet. Eg: {"ticker":"APPL","total_assets":3500000,"total_liabilities":2800000,"shareholders_equity":7000000,"cash_and_equivalents":5000000,"total_debt":1000000}
    """

    ticker=ticker.upper()
    mock_balance_sheet={
        "ticker":ticker,
        "total_assets":9800000,
        "total_liabilities":2800000,
        "shareholders_equity":7000000,
        "cash_and_equivalents":5000000,
        "total_debt":1000000
    }

    return mock_balance_sheet


@tool
def get_cash_flow_statement(ticker:str)->dict:
    """
    Retrieves company's cash flow statement

    Args:
        ticker: ticker symbol for a company whose stock analysis query is searching. Eg:NVDA, APPL.
    
    Returns:
        Dictionary containing company's balance sheet. Eg: {"ticker":"APPL","operating_cash_flow":1100000,"investing_cash_flow": -2000000,"financing_cash_flow": -7000000,"capital_expenditure": 1500000,"free_cash_flow": 9500000}
    """

    ticker=ticker.upper()
    mock_cash_flow={
        "ticker":ticker,
        "operating_cash_flow":1100000,
        "investing_cash_flow": -2000000,
        "financing_cash_flow": -7000000,
        "capital_expenditure": 700000,
        "free_cash_flow": 400000
    }

    return mock_cash_flow


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