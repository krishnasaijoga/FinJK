
from unittest.mock import MagicMock, patch

import pytest

from agents.valuation_agent import (
    DCFValuation,
    PeerValuation,
    ValuationAnalysisOutput,
    ValuationMultiples,
    create_valuation_analysis_agent,
)
from tools.valuation_tools import (
    get_peer_multiples,
    run_dcf_model,
)


# FIXTURE: Mock Valuation Output
@pytest.fixture
def mock_valuation_output():
    return ValuationAnalysisOutput(
        ticker="APPL",
        valuation_status="SLIGHTLY UNDERVALUED",

        dcf_valuation=DCFValuation(
            enterprise_value=1800000000000,
            equity_value=1700000000000,
            intrinsic_value_per_share=113.33,
            current_market_price=100.00,
            upside_percentage=13.33,
        ),

        peer_valuation=PeerValuation(
            peer_tickers=["MSFT", "GOOGL", "AMZN"],

            company_multiples=ValuationMultiples(
                pe_ratio=25.0,
                ev_ebitda=16.0,
                ps_ratio=4.5,
            ),

            peer_average_multiples=ValuationMultiples(
                pe_ratio=28.0,
                ev_ebitda=18.0,
                ps_ratio=5.0,
            ),

            relative_valuation="Trading at a discount",

            summary=(
                "The company trades at lower valuation "
                "multiples than its selected peers."
            ),
        ),

        key_findings=[
            "DCF indicates 13.33% potential upside",
            "The company trades below peer average multiples",
        ],

        summary=(
            "Based on mock valuation data, the stock appears "
            "slightly undervalued, with a 13.33% estimated "
            "DCF upside and a discount relative to peers."
        ),

        data_status=True,
    )


# TEST 1: Agent Creation
@patch("agents.valuation_agent.Agent")
@patch("agents.valuation_agent.OpenAIModel")
def test_create_valuation_agent(
    mock_model,
    mock_agent,
    monkeypatch,
):
    monkeypatch.setenv("GROQ_API_KEY", "fake-test-key")

    agent = create_valuation_analysis_agent()

    mock_model.assert_called_once()
    mock_agent.assert_called_once()

    assert agent is mock_agent.return_value


# TEST 2: Tool Registration
@patch("agents.valuation_agent.Agent")
@patch("agents.valuation_agent.OpenAIModel")
def test_valuation_agent_tools(
    mock_model,
    mock_agent,
    monkeypatch,
):
    monkeypatch.setenv("GROQ_API_KEY", "fake-test-key")

    create_valuation_analysis_agent()

    kwargs = mock_agent.call_args.kwargs

    assert kwargs["name"] == "ValuationAgent"

    assert kwargs["tools"] == [
        run_dcf_model,
        get_peer_multiples,
    ]

    assert kwargs["structured_output_model"] is ValuationAnalysisOutput


# TEST 3: Mock Agent Response
@patch("agents.valuation_agent.Agent")
@patch("agents.valuation_agent.OpenAIModel")
def test_valuation_agent_response(
    mock_model,
    mock_agent,
    mock_valuation_output,
    monkeypatch,
):
    monkeypatch.setenv("GROQ_API_KEY", "fake-test-key")

    fake_result = MagicMock()
    fake_result.structured_output = mock_valuation_output

    fake_agent = mock_agent.return_value
    fake_agent.return_value = fake_result

    agent = create_valuation_analysis_agent()

    result = agent("Give Valuation analysis for ticker APPL.")
    output = result.structured_output

    assert isinstance(output, ValuationAnalysisOutput)
    assert output.ticker == "APPL"
    assert output.valuation_status == "SLIGHTLY UNDERVALUED"
    assert output.data_status is True

    assert output.dcf_valuation is not None
    assert output.dcf_valuation.intrinsic_value_per_share == 113.33
    assert output.dcf_valuation.upside_percentage == 13.33

    assert output.peer_valuation is not None
    assert output.peer_valuation.company_multiples.pe_ratio == 25.0
    assert output.peer_valuation.peer_average_multiples.pe_ratio == 28.0

    assert len(output.peer_valuation.peer_tickers) == 3
    assert len(output.key_findings) == 2

    fake_agent.assert_called_once_with(
        "Give Valuation analysis for ticker APPL."
    )


# TEST 4: Invalid Valuation Status
def test_invalid_valuation_status(mock_valuation_output):
    data = mock_valuation_output.model_dump()

    data["valuation_status"] = "VERY CHEAP"

    with pytest.raises(ValueError):
        ValuationAnalysisOutput(**data)


# TEST 5: Invalid DCF Metric
def test_invalid_dcf_metric(mock_valuation_output):
    data = mock_valuation_output.model_dump()

    data["dcf_valuation"]["enterprise_value"] = "invalid"

    with pytest.raises(ValueError):
        ValuationAnalysisOutput(**data)


# TEST 6: Invalid Peer Multiple
def test_invalid_peer_multiple(mock_valuation_output):
    data = mock_valuation_output.model_dump()

    data["peer_valuation"]["company_multiples"]["pe_ratio"] = "high"

    with pytest.raises(ValueError):
        ValuationAnalysisOutput(**data)


# TEST 7: Missing DCF Fields
def test_nullable_dcf_metric(mock_valuation_output):
    data = mock_valuation_output.model_dump()

    data["dcf_valuation"]["intrinsic_value_per_share"] = None

    result = ValuationAnalysisOutput(**data)

    assert result.dcf_valuation is not None
    assert result.dcf_valuation.intrinsic_value_per_share is None


# TEST 8: Empty Key Findings
def test_empty_key_findings(mock_valuation_output):
    data = mock_valuation_output.model_dump()

    data["key_findings"] = []

    result = ValuationAnalysisOutput(**data)

    assert result.key_findings == []


# TEST 9: Missing Data Status
def test_data_status_false(mock_valuation_output):
    data = mock_valuation_output.model_dump()

    data["data_status"] = False
    data["valuation_status"] = "UNKNOWN"

    result = ValuationAnalysisOutput(**data)

    assert result.data_status is False
    assert result.valuation_status == "UNKNOWN"
