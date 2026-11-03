
import logging

from bedrock_agentcore import BedrockAgentCoreApp
from agents.orchestrator import run_fundamental_analysis

logger = logging.getLogger(__name__)

app = BedrockAgentCoreApp()


@app.entrypoint
def invoke(payload: dict):
    ticker = str(payload.get("ticker", "")).strip().upper()

    if not ticker:
        return {
            "status": "error",
            "message": "Ticker is required"
        }

    logger.info("Starting FinJK analysis for %s", ticker)

    try:
        report = run_fundamental_analysis(ticker)

        return {
            "status": "success",
            "ticker": ticker,
            "report": report.model_dump(mode="json")
        }

    except Exception:
        logger.exception("FinJK analysis failed")

        return {
            "status": "error",
            "message": "Financial analysis failed"
        }


if __name__ == "__main__":
    app.run()