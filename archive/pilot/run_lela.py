"""Runs `python -m lela.cli ...` with one change: LELA does not retry when the LLM API answers 429 (rate limit)
and silently leaves the mention unlinked. Here a 429 is retried after the pause the server asks for.
Usage (from the LELA folder, with its python):  python pilot/run_lela.py --config c.json --input in.txt --output out.jsonl
"""
import os
import runpy
import sys
import time

import requests

sys.path.insert(0, os.getcwd())    # run it from the LELA folder
_post = requests.post


def post_with_retry(*args, **kwargs):
    for attempt in range(10):
        response = _post(*args, **kwargs)
        if response.status_code != 429:
            return response
        wait = float(response.headers.get("retry-after") or 0) or 5 * (attempt + 1)
        print(f"429, waiting {wait:.0f}s", file=sys.stderr)
        time.sleep(min(wait + 1, 90))
    return response


requests.post = post_with_retry
sys.argv = ["lela.cli"] + sys.argv[1:]
runpy.run_module("lela.cli", run_name="__main__")
