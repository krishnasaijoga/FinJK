
from unittest.mock import MagicMock, patch

import pytest

from agents.risk_agent import (
    RiskAnalysisOutput,
    create_risk_analysis_agent,
)
from tools.financial_tools import (
    compute_key_ratios,
    get_balance_sheet,
)
from tools.news_tools import search_recent_news


@pytest.fixture
def mock_risk_output():
    """Return a valid mocked risk analysis response."""

    return RiskAnalysisOutput(
        ticker="APPL",
        overall_risk="ELEVATED",
        risk_score=0.65,

        financial_risks=[
            "Elevated debt-to-equity ratio",
            "Weak free cash flow generation",
        ],

        business_risks=[
            "Increasing market competition",
        ],

        regulatory_risks=[
            "Potential regulatory restrictions",
        ],

        key_risk_metrics={
            "debt_to_equity": 2.5,
            "net_profit_margin": 0.08,
            "free_cash_flow_margin": -0.05,
        },

        summary=(
            "The company faces elevated financial risks "
            "due to leverage and weak cash generation, "
            "alongside competitive and regulatory pressures."
        ),

        data_status=True,
    )


# TEST 1: Agent Creation
@patch("agents.risk_agent.Agent")
@patch("agents.risk_agent.OpenAIModel")
def test_create_risk_analysis_agent(
    mock_model,
    mock_agent,
    monkeypatch,
):
    monkeypatch.setenv("GROQ_API_KEY", "fake-test-key")

    agent = create_risk_analysis_agent()

    mock_model.assert_called_once()
    mock_agent.assert_called_once()

    assert agent is mock_agent.return_value


# TEST 2: Tool Registration
@patch("agents.risk_agent.Agent")
@patch("agents.risk_agent.OpenAIModel")
def test_risk_agent_tools(
    mock_model,
    mock_agent,
    monkeypatch,
):
    monkeypatch.setenv("GROQ_API_KEY", "fake-test-key")

    create_risk_analysis_agent()

    kwargs = mock_agent.call_args.kwargs

    assert kwargs["name"] == "RiskAnalysisAgent"

    assert kwargs["tools"] == [
        search_recent_news,
        get_balance_sheet,
        compute_key_ratios,
    ]

    assert kwargs["structured_output_model"] is RiskAnalysisOutput


# TEST 3: Mock Agent Response
@patch("agents.risk_agent.Agent")
@patch("agents.risk_agent.OpenAIModel")
def test_risk_agent_response(
    mock_model,
    mock_agent,
    mock_risk_output,
    monkeypatch,
):
    monkeypatch.setenv("GROQ_API_KEY", "fake-test-key")

    fake_result = MagicMock()
    fake_result.structured_output = mock_risk_output

    fake_agent = mock_agent.return_value
    fake_agent.return_value = fake_result

    agent = create_risk_analysis_agent()

    result = agent("Analyze risk for ticker APPL.")

    output = result.structured_output

    assert isinstance(output, RiskAnalysisOutput)
    assert output.ticker == "APPL"
    assert output.overall_risk == "ELEVATED"
    assert output.risk_score == pytest.approx(0.65)
    assert output.data_status is True

    assert len(output.financial_risks) == 2
    assert len(output.business_risks) == 1
    assert len(output.regulatory_risks) == 1

    assert output.key_risk_metrics["debt_to_equity"] == 2.5
    assert output.key_risk_metrics["net_profit_margin"] == 0.08

    fake_agent.assert_called_once_with(
        "Analyze risk for ticker APPL."
    )


# TEST 4: Invalid Risk Classification
def test_invalid_risk_level(mock_risk_output):
    data = mock_risk_output.model_dump()

    data["overall_risk"] = "EXTREME"

    with pytest.raises(ValueError):
        RiskAnalysisOutput(**data)


# TEST 5: Risk Score Out of Range
@pytest.mark.parametrize("score", [-0.1, 1.5])
def test_invalid_risk_score(mock_risk_output, score):
    data = mock_risk_output.model_dump()

    data["risk_score"] = score

    with pytest.raises(ValueError):
        RiskAnalysisOutput(**data)


# TEST 6: Nullable Risk Score
def test_nullable_risk_score(mock_risk_output):
    data = mock_risk_output.model_dump()

    data["risk_score"] = None

    result = RiskAnalysisOutput(**data)

    assert result.risk_score is None


# TEST 7: Invalid Risk Metrics
def test_invalid_risk_metrics(mock_risk_output):
    data = mock_risk_output.model_dump()

    data["key_risk_metrics"] = {
        "debt_to_equity": "very high"
    }

    with pytest.raises(ValueError):
        RiskAnalysisOutput(**data)


# TEST 8: Empty Risk Categories
def test_empty_risk_categories(mock_risk_output):
    data = mock_risk_output.model_dump()

    data["financial_risks"] = []
    data["business_risks"] = []
    data["regulatory_risks"] = []

    result = RiskAnalysisOutput(**data)

    assert result.financial_risks == []
    assert result.business_risks == []
    assert result.regulatory_risks == []
