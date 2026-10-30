import logging
import time

from dotenv import load_dotenv

from agents.orchestrator import run_fundamental_analysis


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def main():
    ticker = "INFY"

    print(f"\nStarting live fundamental analysis: {ticker}")

    start = time.perf_counter()

    try:
        report = run_fundamental_analysis(ticker)

        duration = time.perf_counter() - start

        print("\n" + "=" * 60)
        print("FINAL INVESTMENT REPORT")
        print("=" * 60)

        print(report.model_dump_json(indent=2))

        print(f"\nExecution time: {duration:.2f} seconds")

        if report.data_status:
            print("\nAnalysis completed with sufficient data.")
        else:
            print("\nAnalysis completed with incomplete data.")

    except Exception:
        logging.exception(
            "Live fundamental analysis failed"
        )
        raise


if __name__ == "__main__":
    main()
