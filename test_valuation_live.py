
from dotenv import load_dotenv

load_dotenv()

from agents.valuation_agent import create_valuation_analysis_agent


def main():
    agent = create_valuation_analysis_agent()

    # Deliberately test with only a ticker.
    # The agent should not run DCF without
    # upstream FinancialDataAgent data.
    prompt = (
        "Analyze stock ticker AAPL. "
        "No FinancialDataAgent output is provided. "
        "Do not fabricate missing financial inputs. "
        "Use peer valuation only if valid peer data "
        "is available. Return the structured output."
    )

    try:
        result = agent(prompt)

        print("\nAgent execution completed.")

        output = result.structured_output

        if output is None:
            print("No structured output returned.")
        else:
            print(output.model_dump_json(indent=2))

    except Exception as exc:
        print(f"\nException type: {type(exc).__name__}")
        print(f"Exception message: {exc}")
        raise


if __name__ == "__main__":
    main()
