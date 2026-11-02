
import json
import logging
import os
import sys

from agents.orchestrator import run_fundamental_analysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    ticker = os.getenv("TICKER", "MSFT").strip().upper()

    logger.info("Starting FinJK analysis for %s", ticker)

    try:
        report = run_fundamental_analysis(ticker)

        print(
            json.dumps(
                {
                    "status": "success",
                    "report": report.model_dump(),
                },
                indent=2,
                default=str,
            ),
            flush=True,
        )

        logger.info("FinJK analysis completed")
    except Exception:
        logger.exception("FinJK analysis Failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
