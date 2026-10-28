from strands import tool


@tool
def run_dcf_model(ticker:str)->dict:
    """
    Estimates Discounted Cash Flow worth of a company today based on its other financial metrics

    Args:
        ticker: ticker symbol for a company whose stock analysis query is searching. Eg:NVDA, APPL.
    
    Returns:
        Dictionary containing company's dcf related metrics. Eg: {"ticker":"APPL","valuation_method": "DCF","enterprise_value": 1800000000000,"equity_value": 1700000000000,"intrinsic_value_per_share": 113.33,"current_market_price": 100.00,"upside_percentage": 13.33,"valuation_status": "Undervalued"}
    """

    ticker=ticker.upper()

    mock_dcf_results={
        "ticker": ticker,
        "valuation_method": "DCF",
        "enterprise_value": 1800000000000,
        "equity_value": 1700000000000,
        "intrinsic_value_per_share": 113.33,
        "current_market_price": 100.00,
        "upside_percentage": 13.33,
        "valuation_status": "Undervalued"
        }
    
    return mock_dcf_results


@tool
def get_peer_multiples(ticker:str)->dict:
    """
    Evaluates and returns whether a companys stock is expensive or cheap compared with similar companies.

    Args:
        ticker: ticker symbol for a company whose stock analysis query is searching. Eg:NVDA, APPL.
    
    Returns:
        Dictionary containing company's dcf related metrics. Eg: {"ticker":"APPL","valuation_method": "DCF","enterprise_value": 1800000000000,"equity_value": 1700000000000,"intrinsic_value_per_share": 113.33,"current_market_price": 100.00,"upside_percentage": 13.33,"valuation_status": "Undervalued"}
    """

    ticker=ticker.upper()
    
    mock_multiples={
        "ticker": ticker,
        "valuation_method": "Peer Multiples",
        "peer_tickers":["MSFT","GOOGL","AMZN"],
        "company_multiples": {
            "pe_ratio": 25.0,
            "ev_ebitda": 16.0,
            "ps_ratio": 4.5
        },
        "peer_average_multiples": {
            "pe_ratio": 28.0,
            "ev_ebitda": 18.0,
            "ps_ratio": 5.0
        },
        "relative_valuation": "Trading at a discount",
        "summary": "The company trades at lower valuation multiples than the selected peer group."
        }
    return mock_multiples