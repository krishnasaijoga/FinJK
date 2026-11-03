
import logging
import os

from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, Field
from hmac import compare_digest

from agents.orchestrator import run_fundamental_analysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FinJK Fundamental Analysis API",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
)


class AnalysisRequest(BaseModel):
    ticker: str = Field(
        min_length=1,
        max_length=20,
        pattern=r"^[A-Za-z0-9.^-]+$"
    )


def verify_api_key(x_api_key: str = Header(default="")):
    expected = os.environ.get("FINJK_API_TOKEN")

    if not expected:
        raise HTTPException(
            status_code=503,
            detail="API authentication not configured"
        )

    if not compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized"
        )


@app.get("/health")
def health():
    return {"status": "healthy", "service": "finjk"}


@app.post("/analyze", dependencies=[Depends(verify_api_key)])
def analyze(request: AnalysisRequest):
    ticker = request.ticker.strip().upper()
    logger.info("Starting analysis for %s", ticker)

    try:
        report = run_fundamental_analysis(ticker)

        logger.info("Completed analysis for %s", ticker)
        return report.model_dump()

    except Exception:
        logger.exception("Analysis failed for %s", ticker)

        raise HTTPException(
            status_code=503,
            detail="Financial analysis temporarily unavailable"
        )
