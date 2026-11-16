"""Client for the Jev API (TypeSafe System One): typed questions about a state, answered with probabilities.

What it does beyond one POST:
* every answer is cached by (model, state, question) in jev/cache/answers.jsonl, so a rerun asks nothing twice;
* questions that share a state go out together, in chunks sized by questions and tokens, and chunks run in parallel;
* 429 and 529 are retried with backoff; any other failure is an error, never an answer;
* the model is pinned to a version, because thresholds do not carry over between versions.
Standard library only. Key: JEV_API_KEY in .env or in the environment.
"""
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
API = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.14.0"          # jev-latest is an alias that moves
CHUNK, TOKENS = 24, 40000     # per request: questions, and estimated tokens (the API allows 64k, and 32k a question)
WORKERS = 6


def load_env(path=os.path.join(HERE, "..", ".env")):
    """NAME=value lines of .env into the environment; what is already set there wins."""
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            name, eq, value = line.strip().partition("=")
            if eq and not name.startswith("#"):
                os.environ.setdefault(name.strip(), value.strip().strip('"'))


load_env()      # JEV_API_KEY here; WIKIPEDIA_CONTACT and YAGO_JSONL for sources.py


class Jev:
    def __init__(self, model=MODEL, cache=os.path.join(HERE, "cache", "answers.jsonl")):
        self.model, self.path = model, cache
        self.lock = threading.Lock()
        self.stats = {"questions": 0, "cached": 0, "requests": 0, "input_tokens": 0}
        self.cache = {}
        if os.path.exists(cache):
            for line in open(cache, encoding="utf-8"):
                row = json.loads(line)
                self.cache[row["k"]] = row["a"]

    def _k(self, state, question):
        blob = json.dumps([self.model, state, question], ensure_ascii=False)  # option order is part of a question
        return hashlib.sha256(blob.encode()).hexdigest()

    def _post(self, state, questions):
        key = os.environ.get("JEV_API_KEY")     # only needed for a question that is not in the cache
        if not key:
            raise RuntimeError("JEV_API_KEY is missing: put it in .env (see .env.example)")
        body = json.dumps({"model": self.model, "state": state, "questions": questions}).encode()
        for attempt in range(8):
            req = urllib.request.Request(API, body, {"Authorization": "Bearer " + key,
                                                     "Content-Type": "application/json",
                                                     "User-Agent": "python-urllib"})
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    out = json.load(r)
                with self.lock:
                    self.stats["requests"] += 1
                    self.stats["input_tokens"] += out.get("usage", {}).get("input_tokens", 0)
                return out["answers"]
            except urllib.error.HTTPError as e:
                if e.code not in (429, 529):
                    raise RuntimeError(f"Jev API {e.code}: {e.read().decode(errors='replace')[:500]}") from None
                time.sleep(min(60, 2 * 2 ** attempt))
            except (urllib.error.URLError, TimeoutError):
                time.sleep(min(60, 2 * 2 ** attempt))
        raise RuntimeError("Jev API: no answer after 8 attempts")

    def ask_many(self, jobs):
        """jobs: [(state, {name: question})] -> [{name: answer}], in the order given."""
        out, groups = [{} for _ in jobs], {}
        for i, (state, questions) in enumerate(jobs):
            for name, q in questions.items():
                k = self._k(state, q)
                self.stats["questions"] += 1
                if k in self.cache:
                    out[i][name] = self.cache[k]
                    self.stats["cached"] += 1
                else:                                   # jobs with the same state share requests
                    groups.setdefault(json.dumps(state, ensure_ascii=False), (state, []))[1].append((i, name, q, k))
        todo = []
        for blob, (state, items) in groups.items():
            chunk, size = [], 0
            for item in items:
                cost = (len(blob) + len(json.dumps(item[2], ensure_ascii=False))) // 3   # tokens, estimated high
                if chunk and (len(chunk) == CHUNK or size + cost > TOKENS):
                    todo.append((state, chunk))
                    chunk, size = [], 0
                chunk.append(item)
                size += cost
            todo.append((state, chunk))

        def run(job):
            state, items = job
            answers = self._post(state, {f"q{n}": q for n, (_, _, q, _) in enumerate(items)})
            with self.lock:
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as f:
                    for n, (i, name, _, k) in enumerate(items):
                        self.cache[k] = out[i][name] = answers[f"q{n}"]
                        f.write(json.dumps({"k": k, "a": answers[f"q{n}"]}, ensure_ascii=False) + "\n")

        if todo:
            with ThreadPoolExecutor(WORKERS) as pool:
                list(pool.map(run, todo))
        return out

    def ask(self, state, questions):
        return self.ask_many([(state, questions)])[0]

    def report(self):
        s = self.stats
        return (f"{s['questions']} questions ({s['cached']} from cache), {s['requests']} requests, "
                f"{s['input_tokens']} input tokens (${s['input_tokens'] * 0.042 / 1e6:.4f})")
