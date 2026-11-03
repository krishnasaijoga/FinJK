from functools import lru_cache

import pandas as pd
import yfinance as yf


class YahooFinanceError(Exception):
    pass


@lru_cache(maxsize=32)
def get_stock(ticker:str)->yf.Ticker:
    ticker=ticker.strip().lower()
    if not ticker:
        raise ValueError("Ticker cannot be empty")
    return yf.Ticker(ticker)


def get_statement(ticker:str,statement:str,period:str="annual")->dict:
    """
    Return the latest reported financial statement as a normalized dictionary
    """
    if period not in ("annual","quarterly"):
        raise ValueError("period must be either 'annual' or 'quarterly'")

    stock=get_stock(ticker)
    frequency="yearly" if period =="annual" else "quarterly"

    try:
        if statement=="income":
            df=stock.get_income_stmt(freq=frequency)
        elif statement=="balance":
            df=stock.get_balance_sheet(freq=frequency)
        elif statement=="cashflow":
            df=stock.get_cashflow(freq=frequency)
        else:
            raise ValueError(f"Unknown statements: {statement}")
    except (ValueError, KeyError, TypeError):
        raise
    except Exception  as exc:
        raise ValueError(f"Failed to fetch {statement} for {ticker}") from exc

    if df is None or df.empty: # type: ignore
        return {
            "ticker":ticker.upper(),
            "period":period,
            "statement":statement,
            "data_status":False,
            "report_date":None,
            "metrics":{}
        }

    latest_date=max(df.columns) # type: ignore
    latest=df[latest_date]

    metrics={}

    for key,value in latest.items():
        metrics[str(key)]=(None if pd.isna(value) else float(value))

    return {
        "ticker":ticker.upper(),
        "period":period,
        "statement":statement,
        "data_status":True,
        "report_date":pd.Timestamp(latest_date).date().isoformat(),
        "metrics":metrics
    }


def get_metric(
        metrics:dict,
        *possible_names:str
)->float|None:
    """Try available Yahoo field names in order."""
    for name in possible_names:
        value=metrics.get(name)
        if value is not None:
            return float(value)

    return None