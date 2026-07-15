import logging
import sys
import uuid

from judge_graph import run_judge
from report import print_debate_report

LOG_PATH = "debate.log"


class _ErrorCountingHandler(logging.Handler):
    """Counts ERROR+ records logged during a run (Gemini/Groq rate-limit
    failures, parse failures, etc.) so main() can print a one-line pointer
    to LOG_PATH instead of staying silent about them."""

    def __init__(self):
        super().__init__(level=logging.ERROR)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        self.count += 1


def _configure_logging() -> _ErrorCountingHandler:
    """Route logging to LOG_PATH instead of the console -- main.py's
    terminal output should be just the formatted report, not raw logger
    noise from real (but non-fatal, degrade-on-failure) API errors.
    Errors are still fully captured on disk, just not printed."""
    file_handler = logging.FileHandler(LOG_PATH)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    error_counter = _ErrorCountingHandler()

    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_counter)
    return error_counter


def main() -> None:
    error_counter = _configure_logging()

    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    run_id = str(uuid.uuid4())
    result = run_judge(ticker, run_id)
    print_debate_report(ticker, run_id, result)

    if error_counter.count:
        print(f"\n({error_counter.count} error(s) logged during this run -- see {LOG_PATH} for details)")


if __name__ == "__main__":
    main()
