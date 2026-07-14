import os

from dotenv import load_dotenv

load_dotenv()

# Named judge_config.py rather than config.py: the installed
# stock-signal-system package owns the top-level `config` module name in
# this venv (see requirements.txt), so this project's own settings live
# under a different name to avoid shadowing/collision.

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
JUDGE_MODEL_NAME = os.getenv("JUDGE_MODEL_NAME", "claude-haiku-4-5-20251001")
