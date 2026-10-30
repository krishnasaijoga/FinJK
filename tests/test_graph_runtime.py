
import asyncio
import time
from unittest.mock import patch

import pytest
from strands.agent.agent_result import AgentResult
from strands.multiagent.base import (
    MultiAgentBase,
    MultiAgentResult,
    NodeResult,
    Status,
)
from strands.telemetry.metrics import EventLoopMetrics

from agents import orchestrator as orch


class FakeSpecialist(MultiAgentBase):
    """
    Deterministic replacement for a specialist agent.

    Uses the real Strands Graph scheduler, but
    never makes an LLM request.
    """

    def __init__(
        self,
        name,
        events,
        delay=0.05,
        result_text=None,
    ):
        super().__init__()
        self.name = name
        self.events = events
        self.delay = delay
        self.result_text = (
            result_text or f"{name} completed"
        )
        self.received_task = None

    async def invoke_async(
        self,
        task,
        invocation_state=None,
        **kwargs,
    ):
        self.received_task = task

        self.events.append(
            (self.name, "start", time.perf_counter())
        )

        await asyncio.sleep(self.delay)

        self.events.append(
            (self.name, "end", time.perf_counter())
        )

        # Fake agent result containing a recognizable
        # text response and no actual LLM interaction.
        agent_result = AgentResult(
            stop_reason="end_turn",
            message={
                "role":"assistant",
                "content":[
                    {"text":self.result_text}
                ]
            },
            metrics=EventLoopMetrics(),
            state={},
            structured_output=None
        )

        NodeResult(
            result=agent_result,
            status=Status.COMPLETED
        )

        node_result=NodeResult(
            result=agent_result,
            status=Status.COMPLETED
        )

        return MultiAgentResult(
            status=Status.COMPLETED,
            results={
                self.name: node_result,
            },
        )


@pytest.fixture
def runtime_setup():
    events = []

    agents = {
        "financial": FakeSpecialist(
            "financial",
            events,
            delay=0.15,
            result_text="FINANCIAL_TOKEN_123 revenue=40000000",
        ),
        "news": FakeSpecialist(
            "news",
            events,
            delay=0.10,
        ),
        "risk": FakeSpecialist(
            "risk",
            events,
            delay=0.12,
        ),
        "valuation": FakeSpecialist(
            "valuation",
            events,
            delay=0.02,
        ),
    }

    return events, agents


def execute_real_graph(agents):
    """
    Patch the specialist factories only.

    create_analysis_graph() still constructs and
    executes a real Strands Graph.
    """

    with (
        patch.object(
            orch,
            "create_financial_data_agent",
            return_value=agents["financial"],
        ),
        patch.object(
            orch,
            "create_news_sentiment_agent",
            return_value=agents["news"],
        ),
        patch.object(
            orch,
            "create_risk_analysis_agent",
            return_value=agents["risk"],
        ),
        patch.object(
            orch,
            "create_valuation_analysis_agent",
            return_value=agents["valuation"],
        ),
    ):
        graph = orch.create_analysis_graph()

        result = graph(
            "Analyze stock ticker APPL."
        )

    return result


def get_time(events, name, event_type):
    return next(
        timestamp
        for agent, event, timestamp in events
        if agent == name and event == event_type
    )


def test_all_four_nodes_execute(runtime_setup):
    events, agents = runtime_setup

    result = execute_real_graph(agents)

    assert result.status == Status.COMPLETED

    assert set(result.results) == {
        "financial",
        "news",
        "risk",
        "valuation",
    }

    for name in agents:
        assert get_time(events, name, "start")
        assert get_time(events, name, "end")


def test_initial_agents_execute_concurrently(
    runtime_setup,
):
    events, agents = runtime_setup

    execute_real_graph(agents)

    names = ["financial", "news", "risk"]

    latest_start = max(
        get_time(events, name, "start")
        for name in names
    )

    earliest_end = min(
        get_time(events, name, "end")
        for name in names
    )

    # All three began before any of them finished.
    assert latest_start < earliest_end


def test_valuation_waits_for_financial(
    runtime_setup,
):
    events, agents = runtime_setup

    execute_real_graph(agents)

    financial_end = get_time(
        events, "financial", "end"
    )

    valuation_start = get_time(
        events, "valuation", "start"
    )

    assert valuation_start >= financial_end


def test_valuation_receives_financial_context(
    runtime_setup,
):
    _events, agents = runtime_setup

    execute_real_graph(agents)

    valuation_input = str(
        agents["valuation"].received_task
    )

    assert "FINANCIAL_TOKEN_123" in valuation_input


def test_every_agent_runs_once(runtime_setup):
    events, agents = runtime_setup

    execute_real_graph(agents)

    for name in agents:
        starts = [
            event
            for event in events
            if event[0] == name
            and event[1] == "start"
        ]

        assert len(starts) == 1
