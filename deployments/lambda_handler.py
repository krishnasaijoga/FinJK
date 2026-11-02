
import json
import logging
import os

import boto3

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def configure_api_credentials():
    secret_id = os.environ["FINJK_SECRET_ID"]

    client = boto3.client(
        "secretsmanager",
        region_name=os.environ.get(
            "AWS_REGION", "ap-southeast-2"
        ),
    )

    response = client.get_secret_value(
        SecretId=secret_id
    )

    credentials = json.loads(
        response["SecretString"]
    )

    os.environ["GROQ_API_KEY"] = credentials["GROQ_API_KEY"]
    os.environ["GNEWS_API_KEY"] = credentials["GNEWS_API_KEY"]

    logger.info("API credentials loaded from Secrets Manager")


def lambda_handler(event, context):
    try:
        ticker = event.get("ticker", "").strip().upper()

        if not ticker:
            return {
                "statusCode": 400,
                "body": json.dumps({
                    "error": "Ticker is required"
                }),
            }

        # Load credentials BEFORE creating agents.
        configure_api_credentials()

        from agents.orchestrator import (
            run_fundamental_analysis
        )

        result = run_fundamental_analysis(ticker)

        return {
            "statusCode": 200,
            "body": result.model_dump_json(),
        }

    except Exception:
        logger.exception(
            "Fundamental Finance Analysis Failed"
        )

        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": "Analysis execution failed"
            }),
        }