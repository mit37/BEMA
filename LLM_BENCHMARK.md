# Workstream C: independent speed/cost benchmark vs a real LLM (ruled out, not approximated)

Date: 2026-09-22. Requested experiment: run the same decisions already in
this repo's test sets (a sample of Noul/Choice/Score examples) through a
real LLM as a text-generation classification task (prompted to emit the
same typed answer), measure real wall-clock latency and real token
cost/pricing, and compare directly against BEMA's own measured latency
(`benchmark_speed.py`, ~1.4ms/example) — the point being to test Jev's
40-200x speed / ~400x cost claim against a real measurement instead of
`benchmark_speed.py`'s existing documented-industry-reference-figure
comparison.

## Result: not run. No usable LLM API is reachable from this sandbox.

This was checked directly, not assumed, in three parts:

**1. Network reachability of a real LLM endpoint.** `api.anthropic.com` IS
network-reachable from this sandbox — the egress proxy's own allowlist
(`__agentproxy/status`) explicitly bypasses it (along with a short list
of other Anthropic infrastructure and package-registry hosts). A raw
`curl` request to `https://api.anthropic.com/v1/messages` returns HTTP
401 (an authentication error from the API itself), not a connection
failure — confirming the network path is genuinely open, unlike
`huggingface.co` and `download.pytorch.org`, which return hard
connection-tunnel rejections (see FINDINGS.md §6).

**2. Credentials.** No usable API key is present in this environment.
Checked directly: `ANTHROPIC_API_KEY`, `CLAUDE_CODE_OAUTH_TOKEN`, and
`ANTHROPIC_AUTH_TOKEN` are all unset. `api.anthropic.com`'s reachability
is this session's own underlying infrastructure path (how this agent
itself is served), not a general-purpose credential handed to scripts
running inside the sandbox for independent, separately-metered API
calls. There is no key available to authenticate a standalone benchmark
request, and using this session's own serving infrastructure to make
self-referential "LLM vs BEMA" comparison calls would not be a clean or
appropriate experiment even if a path to attempt it existed — it would
not represent an arm's-length, real-world LLM API call the way a
customer's own API key and billing would.

**3. Alternatives.** Checked for a local LLM server (Ollama on
`localhost:11434`, a generic server on `127.0.0.1:8080`) — neither is
running. Checked for the `openai` or `anthropic` Python SDKs being
pre-installed with usable default credentials — neither package is
installed at all.

## What this means for the existing speed claim

`benchmark_speed.py`'s existing number (mean 1.4ms, single example, all
three typed heads, one forward pass, CPU-only) stands as a real,
directly-measured property of BEMA's own model. It was never in
question. What remains unmeasured — and, per this project's rule against
faking or approximating an unreachable experiment, stays unmeasured
rather than estimated — is the OTHER side of the ratio: real wall-clock
latency and real metered cost for the equivalent decision made by an
actual LLM API call in this environment. FINDINGS.md's value-proposition
table already flags "40-200x speed advantage" as this repo's
weakest-evidenced claim, for exactly this reason; this workstream
confirms that gap remains, with the specific reachability/credential
checks now on record rather than assumed.

## What would resolve this

Someone with an API key for any hosted LLM (their own account, their own
billing) could run this experiment directly against this repo:
1. Sample N examples from `data/prepared_multitask.pkl`'s test splits
   (already real, already labeled).
2. Prompt the LLM to emit the same typed answer (e.g. `{"is_spam": true,
   "confidence": 0.9}` for Noul) for each, using their real API key.
3. Measure real wall-clock round-trip latency and real token-based cost
   from the provider's own pricing.
4. Compare directly to `python3 benchmark_speed.py`'s output on the same
   hardware class.

This is a well-defined, five-line script away from being run — it is
reported here as blocked by sandbox access, not by any remaining design
uncertainty.
