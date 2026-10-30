import logging

from agents.orchestrator import run_fundamental_analysis

if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)

    ticker="APPL"

    try:
        report=run_fundamental_analysis(ticker)

        print(report.model_dump_json(indent=2))
    except Exception:
        logging.exception("Fundamental analysis workflow failed")
        raise