
import pytest
from unittest.mock import patch, MagicMock

from agents.report_writer_agent import (
    create_report_writer_agent,
    generate_investment_report,
    ReportWriterOutput,
)


@pytest.fixture
def mock_specialist_outputs():
    """Create mock inputs for all four specialist agents."""

    financial = MagicMock()
    financial.ticker = "APPL"
    financial.model_dump.return_value = {
        "ticker": "APPL",
        "overall_company_health": "STRONG",
        "net_profit_margin": 0.2375,
        "data_status": True,
    }

    news = MagicMock()
    news.ticker = "APPL"
    news.model_dump.return_value = {
        "ticker": "APPL",
        "overall_sentiment": "NEUTRAL",
        "articles_analysed": 3,
        "data_status": True,
    }

    risk = MagicMock()
    risk.ticker = "APPL"
    risk.model_dump.return_value = {
        "ticker": "APPL",
        "overall_risk": "ELEVATED",
        "risk_score": 0.65,
        "data_status": True,
    }

    valuation = MagicMock()
    valuation.ticker = "APPL"
    valuation.model_dump.return_value = {
        "ticker": "APPL",
        "valuation_status": "SLIGHTLY UNDERVALUED",
        "data_status": True,
    }

    return financial, news, risk, valuation


@pytest.fixture
def mock_report_output():
    """Create a valid ReportWriterOutput."""

    return ReportWriterOutput(
        ticker="APPL",
        investment_recommendation="HOLD",
        key_reasons=[
            "Strong financial fundamentals",
            "Elevated business and regulatory risks",
            "Modest valuation upside",
        ],
        supporting_evidence={
            "financial": "Net profit margin of 23.75%",
            "sentiment": "Neutral recent news sentiment",
            "risk": "Elevated overall risk",
            "valuation": "DCF indicates modest upside",
        },
        summary=(
            "The company demonstrates strong profitability "
            "but faces elevated business and regulatory risks. "
            "Its valuation suggests modest upside, while "
            "neutral sentiment warrants caution."
        ),
        data_status=True,
    )


# TEST 1: Agent Creation
@patch("agents.report_writer_agent.Agent")
@patch("agents.report_writer_agent.OpenAIModel")
def test_create_report_writer_agent(
    mock_model,
    mock_agent,
    monkeypatch,
):
    monkeypatch.setenv("GROQ_API_KEY", "fake-test-key")

    agent = create_report_writer_agent()

    mock_model.assert_called_once()
    mock_agent.assert_called_once()

    assert agent is mock_agent.return_value


# TEST 2: Agent Configuration
@patch("agents.report_writer_agent.Agent")
@patch("agents.report_writer_agent.OpenAIModel")
def test_report_writer_agent_configuration(
    mock_model,
    mock_agent,
    monkeypatch,
):
    monkeypatch.setenv("GROQ_API_KEY", "fake-test-key")

    create_report_writer_agent()

    kwargs = mock_agent.call_args.kwargs

    assert kwargs["name"] == "ReportWriterAgent"
    assert kwargs["tools"] == []
    assert kwargs["structured_output_model"] is ReportWriterOutput
    assert kwargs["system_prompt"]


# TEST 3: Report Generation With Mock Outputs
@patch("agents.report_writer_agent.create_report_writer_agent")
def test_generate_investment_report(
    mock_create_agent,
    mock_specialist_outputs,
    mock_report_output,
):
    financial, news, risk, valuation = mock_specialist_outputs

    fake_result = MagicMock()
    fake_result.structured_output = mock_report_output

    fake_agent = mock_create_agent.return_value
    fake_agent.return_value = fake_result

    report = generate_investment_report(
        financial_data=financial,
        news_data=news,
        risk_data=risk,
        valuation_data=valuation,
    )

    assert isinstance(report, ReportWriterOutput)
    assert report.ticker == "APPL"
    assert report.investment_recommendation == "HOLD"
    assert report.data_status is True

    assert len(report.key_reasons) == 3
    assert len(report.supporting_evidence) == 4

    fake_agent.assert_called_once()

    # Verify that the input sent to the agent
    # contains all four specialist outputs.
    prompt = fake_agent.call_args.args[0]

    assert "financial_analysis" in prompt
    assert "news_sentiment_analysis" in prompt
    assert "risk_analysis" in prompt
    assert "valuation_analysis" in prompt


# TEST 4: Ticker Mismatch
@patch("agents.report_writer_agent.create_report_writer_agent")
def test_ticker_mismatch(
    mock_create_agent,
    mock_specialist_outputs,
):
    financial, news, risk, valuation = mock_specialist_outputs

    risk.ticker = "NVDA"

    with pytest.raises(ValueError, match="Ticker mismatch"):
        generate_investment_report(
            financial_data=financial,
            news_data=news,
            risk_data=risk,
            valuation_data=valuation,
        )

    mock_create_agent.assert_not_called()


# TEST 5: Missing Structured Output
@patch("agents.report_writer_agent.create_report_writer_agent")
def test_missing_structured_output(
    mock_create_agent,
    mock_specialist_outputs,
):
    financial, news, risk, valuation = mock_specialist_outputs

    fake_result = MagicMock()
    fake_result.structured_output = None

    mock_create_agent.return_value.return_value = fake_result

    with pytest.raises(
        ValueError,
        match="no structured output",
    ):
        generate_investment_report(
            financial_data=financial,
            news_data=news,
            risk_data=risk,
            valuation_data=valuation,
        )


# TEST 6: Incorrect Ticker From Report Writer
@patch("agents.report_writer_agent.create_report_writer_agent")
def test_incorrect_report_ticker(
    mock_create_agent,
    mock_specialist_outputs,
    mock_report_output,
):
    financial, news, risk, valuation = mock_specialist_outputs

    incorrect_output = mock_report_output.model_copy(
        update={"ticker": "NVDA"}
    )

    fake_result = MagicMock()
    fake_result.structured_output = incorrect_output

    mock_create_agent.return_value.return_value = fake_result

    with pytest.raises(
        ValueError,
        match="incorrect ticker",
    ):
        generate_investment_report(
            financial_data=financial,
            news_data=news,
            risk_data=risk,
            valuation_data=valuation,
        )


# TEST 7: Invalid Investment Recommendation
def test_invalid_recommendation(mock_report_output):
    data = mock_report_output.model_dump()

    data["investment_recommendation"] = "MAYBE BUY"

    with pytest.raises(ValueError):
        ReportWriterOutput(**data)


# TEST 8: Invalid Supporting Evidence
def test_invalid_supporting_evidence(mock_report_output):
    data = mock_report_output.model_dump()

    data["supporting_evidence"] = {
        "financial": ["Invalid list instead of string"]
    }

    with pytest.raises(ValueError):
        ReportWriterOutput(**data)


# TEST 9: Unknown Recommendation
def test_unknown_recommendation(mock_report_output):
    data = mock_report_output.model_dump()

    data["investment_recommendation"] = "UNKNOWN"
    data["data_status"] = False

    result = ReportWriterOutput(**data)

    assert result.investment_recommendation == "UNKNOWN"
    assert result.data_status is False
