import logging
import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from strands import Agent
from strands.models.openai import OpenAIModel

from strands.multiagent import GraphBuilder
from strands.multiagent.graph import GraphState
from strands.multiagent.base import Status


from agents.report_writer_agent import ReportWriterOutput
from agents.financial_data_agent import create_financial_data_agent, FinancialDataOutput
from agents.news_sentiment_agent import create_news_sentiment_agent, NewsSentimentOutput
from agents.valuation_agent import create_valuation_analysis_agent, ValuationAnalysisOutput
from agents.risk_agent import create_risk_analysis_agent, RiskAnalysisOutput
from agents.report_writer_agent import ReportWriterOutput, generate_investment_report


load_dotenv()  # reads GROQ_API_KEY from a .env file, if present

logger=logging.getLogger(__name__)

MODEL_ID = os.getenv("GROQ_MODEL_ID", "openai/gpt-oss-120b")


def all_dependencies_complete(required_nodes:list[str]):
    """Wait untill all specified nodes have completed."""

    def check(state:GraphState)->bool:
        return all(
            node_id in state.results
            and state.results[node_id].status==Status.COMPLETED
            for node_id in required_nodes
        )

    return check


def create_analysis_graph():
    builder=GraphBuilder()

    builder.add_node(create_financial_data_agent(),"financial")

    builder.add_node(create_news_sentiment_agent(),"news")

    builder.add_node(create_risk_analysis_agent(),"risk")

    builder.add_node(create_valuation_analysis_agent(),"valuation")

    # builder.add_node(create_report_writer_agent(),"report writer")

    builder.set_entry_point("financial")
    builder.set_entry_point("news")
    builder.set_entry_point("risk")

    builder.add_edge("financial","valuation")

    # required_nodes=[
    #     "financial",
    #     "news",
    #     "risk",
    #     "valuation"
    # ]

    # wait_for_all=all_dependencies_complete(required_nodes)

    # for node in required_nodes:
    #     builder.add_edge(
    #         node,
    #         "report writer",
    #         condition=wait_for_all
    #     )

    builder.set_max_node_executions(4)

    return builder.build()


def get_validated_output(
        graph_result,
        node_id:str,
        expected_model:type[BaseModel]
):
    node=graph_result.results.get(node_id)

    if node is None:
        raise ValueError(f"{node_id} did not produce a result")

    if node.status != Status.COMPLETED:
        raise RuntimeError(f"{node_id} did not complete successfully")

    agent_result=node.result

    if agent_result is None:
        raise ValueError(f"{node_id} returned an empty result")

    output=getattr(agent_result,"structured_output",None)

    if output is None:
        raise ValueError(f"{node_id} returned no structured output")

    return expected_model.model_validate(output)


def run_fundamental_analysis(ticker:str)->ReportWriterOutput:
    ticker=ticker.strip().upper()

    if not ticker:
        raise ValueError("Ticker cannot be empty")

    logger.info("Starting fundamental analysis for %s",ticker)

    graph=create_analysis_graph()

    graph_result=graph(f"Analyze stock ticker {ticker}.")

    if graph_result.status != Status.COMPLETED:
        raise RuntimeError(f"Graph execution failed: {graph_result.status}")

    financial_output=get_validated_output(graph_result,"financial",FinancialDataOutput)
    news_output=get_validated_output(graph_result,"news",NewsSentimentOutput)
    risk_output=get_validated_output(graph_result,"risk",RiskAnalysisOutput)
    valuation_output=get_validated_output(graph_result,"valuation",ValuationAnalysisOutput)

    outputs=[
        financial_output,
        news_output,
        risk_output,
        valuation_output
    ]

    for output in outputs:
        if output.ticker.upper() != ticker: # type: ignore
            raise ValueError(f"Ticker mismatch: expected {ticker}, received {output.ticker}") # type: ignore

    incomplete=[
        type(output).__name__
        for output in outputs if not output.data_status # type: ignore
    ]

    if incomplete:
        logger.warning("Incomplete Specialist data: %s",incomplete)

    report=generate_investment_report(
        financial_data=financial_output, # type: ignore
        news_data=news_output, # type: ignore
        risk_data=risk_output, # type: ignore
        valuation_data=valuation_output # type: ignore
    )   # invokes ReportWriterAgent

    if incomplete:
        report=report.model_copy(
            update={
                "investment_recommendation":"UNKNOWN",
                "data_status":False
            }
        )
    logger.info("Fundamental analysis completed for %s",ticker)

    return report