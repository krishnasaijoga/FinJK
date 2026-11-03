
import json
import logging
import os

import boto3
from bedrock_agentcore.runtime import BedrockAgentCoreApp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_api_keys():
    """Load FinJK credentials from AWS Secrets Manager."""

    if os.getenv("GROQ_API_KEY") and os.getenv("GNEWS_API_KEY"):
        return

    region = os.getenv("AWS_REGION", "ap-southeast-2")

    client = boto3.client(
        "secretsmanager",
        region_name=region
    )

    response = client.get_secret_value(
        SecretId=os.getenv("FINJK_SECRET_ID", "finjk/api-keys")
    )

    secrets = json.loads(response["SecretString"])

    os.environ.setdefault(
        "GROQ_API_KEY",
        secrets["GROQ_API_KEY"]
    )

    os.environ.setdefault(
        "GNEWS_API_KEY",
        secrets["GNEWS_API_KEY"]
    )


# Configure credentials before importing FinJK agents.
load_api_keys()

from agents.orchestrator import run_fundamental_analysis

app = BedrockAgentCoreApp()


@app.entrypoint
def invoke(payload: dict, context):
    ticker = payload.get("ticker", "")

    if not isinstance(ticker, str) or not ticker.strip():
        return {
            "status": "error",
            "message": "A valid ticker is required"
        }

    ticker = ticker.strip().upper()

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
            "message": "Analysis failed"
        }


if __name__ == "__main__":
    app.run()
