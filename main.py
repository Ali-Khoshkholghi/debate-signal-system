import sys
import uuid

from judge_graph import run_judge


def main() -> None:
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    run_id = str(uuid.uuid4())
    result = run_judge(ticker, run_id)
    print(result)


if __name__ == "__main__":
    main()
