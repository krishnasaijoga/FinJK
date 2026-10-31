
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from strands.multiagent.base import Status

from agents import orchestrator as orch
from agents.financial_data_agent import FinancialDataOutput
from agents.news_sentiment_agent import NewsSentimentOutput
from agents.report_writer_agent import ReportWriterOutput
from agents.risk_agent import RiskAnalysisOutput
from agents.valuation_agent import ValuationAnalysisOutput

from agents.orchestrator import get_validated_output


# --------------------------------------------------
# FIXTURES
# --------------------------------------------------

@pytest.fixture
def specialist_outputs():
    """
    Mock specialist results.

    model_construct() avoids having to repeat every
    financial field in orchestration tests.

    These objects are deliberately partial, so they
    should not be used to test financial calculations.
    """
    return {
        "financial": FinancialDataOutput.model_construct(
            ticker="APPL",
            data_status=True,
        ),
        "news": NewsSentimentOutput.model_construct(
            ticker="APPL",
            data_status=True,
        ),
        "risk": RiskAnalysisOutput.model_construct(
            ticker="APPL",
            data_status=True,
        ),
        "valuation": ValuationAnalysisOutput.model_construct(
            ticker="APPL",
            data_status=True,
        ),
    }


@pytest.fixture
def graph_result(specialist_outputs):
    """Fake completed Strands GraphResult."""
    results = {}

    for name, output in specialist_outputs.items():
        agent_result = SimpleNamespace(
            structured_output=output
        )

        results[name] = SimpleNamespace(
            status=Status.COMPLETED,
            result=agent_result,
        )

    return SimpleNamespace(
        status=Status.COMPLETED,
        results=results,
    )


@pytest.fixture
def final_report():
    return ReportWriterOutput(
        ticker="APPL",
        investment_recommendation="HOLD",
        key_reasons=[
            "Strong financial performance",
            "Moderate investment risk",
        ],
        supporting_evidence={
            "financial": "Profitable operations",
            "sentiment": "Neutral news sentiment",
            "risk": "Moderate risk",
            "valuation": "Near fair value",
        },
        summary="APPL receives a HOLD recommendation.",
        data_status=True,
    )


# --------------------------------------------------
# UNIT TESTS: GRAPH CONFIGURATION
# --------------------------------------------------

def test_graph_contains_four_specialist_agents():
    with (
        patch.object(orch, "GraphBuilder") as builder_cls,
        patch.object(
            orch, "create_financial_data_agent"
        ) as financial,
        patch.object(
            orch, "create_news_sentiment_agent"
        ) as news,
        patch.object(
            orch, "create_risk_analysis_agent"
        ) as risk,
        patch.object(
            orch, "create_valuation_analysis_agent"
        ) as valuation,
    ):
        orch.create_analysis_graph()

        builder = builder_cls.return_value

        assert builder.add_node.call_count == 4

        builder.add_node.assert_any_call(
            financial.return_value, "financial"
        )
        builder.add_node.assert_any_call(
            news.return_value, "news"
        )
        builder.add_node.assert_any_call(
            risk.return_value, "risk"
        )
        builder.add_node.assert_any_call(
            valuation.return_value, "valuation"
        )

        builder.build.assert_called_once()


def test_three_parallel_entry_points():
    with (
        patch.object(orch, "GraphBuilder") as builder_cls,
        patch.object(orch, "create_financial_data_agent"),
        patch.object(orch, "create_news_sentiment_agent"),
        patch.object(orch, "create_risk_analysis_agent"),
        patch.object(orch, "create_valuation_analysis_agent"),
    ):
        orch.create_analysis_graph()

        builder = builder_cls.return_value

        actual = {
            call.args[0]
            for call in builder.set_entry_point.call_args_list
        }

        assert actual == {"financial", "news", "risk"}


def test_valuation_depends_on_financial():
    with (
        patch.object(orch, "GraphBuilder") as builder_cls,
        patch.object(orch, "create_financial_data_agent"),
        patch.object(orch, "create_news_sentiment_agent"),
        patch.object(orch, "create_risk_analysis_agent"),
        patch.object(orch, "create_valuation_analysis_agent"),
    ):
        orch.create_analysis_graph()

        builder = builder_cls.return_value

        builder.add_edge.assert_called_once_with(
            "financial", "valuation"
        )


# --------------------------------------------------
# UNIT TESTS: STRUCTURED OUTPUT VALIDATION
# --------------------------------------------------

@pytest.mark.parametrize(
    "node_id,model",
    [
        ("financial", FinancialDataOutput),
        ("news", NewsSentimentOutput),
        ("risk", RiskAnalysisOutput),
        ("valuation", ValuationAnalysisOutput),
    ],
)
def test_extract_specialist_output(
    graph_result, node_id, model
):
    output = orch.get_validated_output(
        graph_result,
        node_id,
        model,
    )

    assert isinstance(output, model)
    assert output.ticker == "APPL" # type: ignore


def test_missing_node_raises_error(graph_result):
    graph_result.results.pop("financial")

    with pytest.raises(
        ValueError, match="did not produce a result"
    ):
        orch.get_validated_output(
            graph_result,
            "financial",
            FinancialDataOutput,
        )


def test_failed_node_raises_error(graph_result):
    graph_result.results["risk"].status = Status.FAILED

    with pytest.raises(
        RuntimeError, match="did not complete successfully"
    ):
        orch.get_validated_output(
            graph_result,
            "risk",
            RiskAnalysisOutput,
        )

# --------------------------------------------------
# UNIT TESTS: ORCHESTRATOR FAILURES
# --------------------------------------------------

def test_empty_ticker_rejected():
    with pytest.raises(
        ValueError, match="Ticker cannot be empty"
    ):
        orch.run_fundamental_analysis("   ")


def test_graph_failure_prevents_report(graph_result):
    graph_result.status = Status.FAILED

    with (
        patch.object(
            orch, "create_analysis_graph"
        ) as create_graph,
        patch.object(
            orch, "generate_investment_report"
        ) as report_writer,
    ):
        create_graph.return_value.return_value = graph_result

        with pytest.raises(
            RuntimeError, match="Graph execution failed"
        ):
            orch.run_fundamental_analysis("APPL")

        report_writer.assert_not_called()


def test_ticker_mismatch_rejected(graph_result):
    graph_result.results[
        "risk"
    ].result.structured_output.ticker = "NVDA"

    with (
        patch.object(
            orch, "create_analysis_graph"
        ) as create_graph,
        patch.object(
            orch, "generate_investment_report"
        ) as report_writer,
    ):
        create_graph.return_value.return_value = graph_result

        with pytest.raises(
            ValueError, match="Ticker mismatch"
        ):
            orch.run_fundamental_analysis("APPL")

        report_writer.assert_not_called()


# --------------------------------------------------
# INTEGRATION TESTS: AGENT OUTPUT HANDOFF
# --------------------------------------------------

def test_complete_orchestration(
    graph_result,
    specialist_outputs,
    final_report,
):
    """
    Integration of all orchestration steps:
    query -> graph results -> validated outputs
    -> ReportWriter -> final output.
    """
    with (
        patch.object(
            orch, "create_analysis_graph"
        ) as create_graph,
        patch.object(
            orch, "generate_investment_report",
            return_value=final_report,
        ) as report_writer,
    ):
        fake_graph = create_graph.return_value
        fake_graph.return_value = graph_result

        result = orch.run_fundamental_analysis("appl")

        # Verify user query was normalized
        fake_graph.assert_called_once_with(
            "Analyze stock ticker APPL."
        )

        # Verify correct handoff to ReportWriter
        report_writer.assert_called_once_with(
            financial_data=specialist_outputs["financial"],
            news_data=specialist_outputs["news"],
            risk_data=specialist_outputs["risk"],
            valuation_data=specialist_outputs["valuation"],
        )

        assert isinstance(result, ReportWriterOutput)
        assert result.ticker == "APPL"
        assert result.investment_recommendation == "HOLD"
        assert result.data_status is True


def test_incomplete_data_produces_unknown(
    graph_result,
    final_report,
):
    """
    A completed agent can still report insufficient
    data. The Orchestrator must recognize that.
    """
    graph_result.results[
        "news"
    ].result.structured_output.data_status = False

    with (
        patch.object(
            orch, "create_analysis_graph"
        ) as create_graph,
        patch.object(
            orch, "generate_investment_report",
            return_value=final_report,
        ) as report_writer,
    ):
        create_graph.return_value.return_value = graph_result

        result = orch.run_fundamental_analysis("APPL")

        report_writer.assert_called_once()

        assert result.investment_recommendation == "UNKNOWN"
        assert result.data_status is False


def test_failed_specialist_blocks_report(
    graph_result,
):
    graph_result.results[
        "financial"
    ].status = Status.FAILED

    with (
        patch.object(
            orch, "create_analysis_graph"
        ) as create_graph,
        patch.object(
            orch, "generate_investment_report"
        ) as report_writer,
    ):
        create_graph.return_value.return_value = graph_result

        with pytest.raises(RuntimeError):
            orch.run_fundamental_analysis("APPL")

        report_writer.assert_not_called()



def test_failed_specialist_node(graph_result):
    graph_result.results["financial"].status = Status.FAILED

    with pytest.raises(
        RuntimeError,
        match="financial did not complete successfully",
    ):
        get_validated_output(
            graph_result,
            "financial",
            FinancialDataOutput,
        )


def test_missing_structured_output(graph_result):
    graph_result.results[
        "financial"
    ].result.structured_output = None

    with pytest.raises(
        ValueError,
        match="financial returned no structured output",
    ):
        get_validated_output(
            graph_result,
            "financial",
            FinancialDataOutput,
        )


def test_invalid_structured_output(graph_result):
    graph_result.results[
        "financial"
    ].result.structured_output = {
        "ticker": None,
        "data_status": "not-a-boolean",
    }

    with pytest.raises(ValidationError):
        get_validated_output(
            graph_result,
            "financial",
            FinancialDataOutput,
        )


def test_missing_specialist_node(graph_result):
    del graph_result.results["financial"]

    with pytest.raises(
        ValueError,
        match="financial did not produce a result",
    ):
        get_validated_output(
            graph_result,
            "financial",
            FinancialDataOutput,
        )


def test_graph_failure_stops_report_generation():
    from types import SimpleNamespace
    from strands.multiagent.base import Status

    failed_result = SimpleNamespace(status=Status.FAILED)

    with (
        patch.object(orch, "create_analysis_graph") as create_graph,
        patch.object(orch, "generate_investment_report") as report_writer,
    ):
        create_graph.return_value.return_value = failed_result

        with pytest.raises(RuntimeError, match="Graph execution failed"):
            orch.run_fundamental_analysis("AAPL")

        report_writer.assert_not_called()