# Best free AI model aggregator

One server in Python 3.13 (`main.py`, standard library only) for
free-model-aggregator-01. It serves the OpenAI Chat Completions API and the
Anthropic Messages API, in JSON and in server-sent events, over the ten
simulated free-tier providers the trial's brief names, and registers its port
and catalog with the env at the trial's start.

## How it works

1. **Read the brief** (`$MAKEFUN_PROMPT_FILE`): the provider directory, the
   sealed quality table and the workload manifest, found by schema in the
   prompt file.
2. **Probe** every provider's catalog route in its own wire (OpenAI,
   Anthropic or Gemini) with its own auth, and build the one catalog
   (`--list` prints it, byte for byte as registered). The catalog's
   `base_url` is the provider's own path (`/p/<id>`), never joined to the
   env URL, so a replay reproduces it exactly.
3. **Route** each request to the best free model that meets its
   requirements: candidates are ranked by the quality table's
   `rating_milli` (highest first, ties by provider id then model id). A
   provider that refuses with 402/429/5xx is skipped and the next
   candidate serves the request. Only free models ever serve one; a paid
   model is never picked.
4. **Translate faithfully**: the request is passed through in the
   provider's wire untouched -- system prompt, user text, images, tools,
   stop sequences, `max_tokens` and JSON mode all survive translation
   exactly, because the provider echoes what it parsed and the client
   compares that echo with what it sent. A JSON request only ever goes to
   a wire that can carry `response_format` (OpenAI, or Gemini via
   `responseMimeType`): the Anthropic wire echoes `json_mode` false.
5. **Answer** in the client's wire, streamed or not as asked, with the
   provider's text whole, its usage counts as its wire reports them, and
   its stop reason mapped (`stop`/`end_turn`/`STOP` are one family).
6. **Poll** `GET /work` and exit 0 when the env says the workload is done.

Every choice is a function of the brief, the requests and the providers'
answers: no wall clock, no unseeded randomness, so a replay leaves the
same state export.

## Run it

```sh
python3 main.py --list   # print the catalog and exit 0
python3 main.py          # serve a trial (needs MAKEFUN_ENV_URL and
                         # MAKEFUN_PROMPT_FILE)
```

## Local trial results (practice instances, sealed scorer)

| trial | score | quality | served | wire | failover | catalog | calls |
|-------|-------|---------|--------|------|----------|---------|-------|
| 0     | 99    | 99      | 100    | 100  | 100      | 100     | 30    |
| 1     | 99    | 99      | 100    | 100  | 100      | 100     | 36    |
| 2     | 99    | 99      | 100    | 100  | 100      | 100     | 36    |

20/20 requests valid on every trial, 233/233 response checks passed each
time. The quality loss is forced: the best model per request is
1475/1458, but the top providers' per-trial budgets and failure schedules
spend out, so a few requests fall back to the next-best free model
(1470/1465/1452).
