# free-model-aggregator-01: the best free model aggregator.
#
# One server (Python 3.13, standard library only) that serves the OpenAI Chat
# Completions API and the Anthropic Messages API, in JSON and in server-sent
# events, over the ten simulated free-tier providers the trial's brief names.
# It reads the brief, probes every provider's catalog route, registers its
# port and catalog with the env, serves the 20 requests the env's client sends
# (one at a time), and exits 0 when the env says the workload is done.
#
# Routing: for each request, rank every free model that meets the request's
# requirements by the sealed quality table's rating (highest first, ties by
# provider id then model id), call the best provider, and on a 402/429/5xx
# fail over to the next. Only free models ever serve a request; a paid model
# is never picked. Every choice is a function of the brief, the requests and
# the providers' answers: no wall clock, no unseeded randomness, so a replay
# leaves the same state export.
#
# Wire fidelity: the provider echoes what it parsed, and the env's client
# compares that echo with what it sent, so a request is passed through in the
# provider's wire untouched -- system prompt, user text, images, tools, stop
# sequences, max_tokens and JSON mode all survive translation exactly. The
# provider's answer text is returned whole, its usage counts as its wire
# reports them, and its stop reason is mapped to the client's wire. A JSON
# request is only ever routed to a wire that can carry response_format
# (OpenAI, or Gemini via responseMimeType): the Anthropic wire echoes
# json_mode false, so it cannot serve one the client asked in JSON.
#
# `python3 main.py --list` prints the catalog document, byte for byte as it
# registers it, and exits 0. Run outside a trial (no brief, no env) it prints
# {"providers":[]} and exits 0.

import json
import os
import sys
import threading
import time
import urllib.parse
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = 1
PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"

# Fixed timestamps, so no reply reads the clock (a replay must match).
FIXED_OPENAI_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"

PROBE_TIMEOUT_S = 5.0
PROVIDER_TIMEOUT_S = 25.0
REGISTER_TIMEOUT_S = 5.0
WORK_TIMEOUT_S = 2.0
POLL_INTERVAL_S = 0.1
MAX_BODY_BYTES = 32 * 1024 * 1024
SSE_DELTA_CHARS = 13

# Stop-reason families, as the env's scorer maps them: 0 stop, 1 length,
# 2 tool call, 3 refusal. A provider only ever answers family 0.
STOP_FAMILY = {
    "stop": 0, "end_turn": 0, "STOP": 0,
    "length": 1, "max_tokens": 1, "MAX_TOKENS": 1,
    "tool_calls": 2, "function_call": 2, "tool_use": 2,
    "content_filter": 3, "refusal": 3, "SAFETY": 3,
}
OPENAI_STOP = {0: "stop", 1: "length", 2: "tool_calls", 3: "content_filter"}
ANTHROPIC_STOP = {0: "end_turn", 1: "max_tokens", 2: "tool_use", 3: "refusal"}


def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


def is_record(value):
    return value is not None and isinstance(value, dict)


def string_of(value):
    return value if isinstance(value, str) else ""


def number_or_none(value):
    return value if isinstance(value, (int, float)) and value == value and abs(value) != float("inf") else None


def bool_or_none(value):
    return value if isinstance(value, bool) else None


def string_list(value):
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []
    return [x for x in value if isinstance(x, str)]


# ---------------------------------------------------------------------------
# The brief: three JSON documents found by schema in the prompt file.
# ---------------------------------------------------------------------------

def _object_end(text, start):
    """Index of the `}` closing the object that opens at `start`, or -1."""
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def documents_in(text):
    """Every JSON object in `text` that carries a string `schema`, in order."""
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except Exception:
                value = None
            if is_record(value) and isinstance(value.get("schema"), str):
                out.append(value)
        start = text.find("{", start + 1)
    return out


def read_brief(path):
    """Return (directory, ratings, workload, error); empty directory on failure."""
    directory = {"schema": PROVIDERS_SCHEMA, "providers": []}
    if not path:
        return directory, None, None, "MAKEFUN_PROMPT_FILE is not set"
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except Exception as exc:
        return directory, None, None, "cannot read %s: %s" % (path, exc)
    docs = documents_in(text)
    directory_doc = None
    for doc in docs:
        if doc.get("schema") == PROVIDERS_SCHEMA and isinstance(doc.get("providers"), list):
            directory_doc = doc
            break
    if directory_doc is None:
        return directory, None, None, "the brief at %s holds no %s document" % (path, PROVIDERS_SCHEMA)
    providers = []
    for raw in directory_doc["providers"]:
        if not is_record(raw):
            continue
        provider_id = string_of(raw.get("id"))
        wire = raw.get("wire")
        if provider_id == "" or wire not in ("openai", "anthropic", "gemini"):
            continue
        auth = raw.get("auth")
        if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
            auth = "bearer"
        base = ""
        for key in ("path", "base_path", "base_url", "base"):
            if string_of(raw.get(key)) != "":
                base = string_of(raw.get(key))
                break
        providers.append({
            "id": provider_id,
            "name": string_of(raw.get("name")) or provider_id,
            "wire": wire,
            "auth": auth,
            "base": base,
            "key_env": string_of(raw.get("key_env")),
            "key": string_of(raw.get("key")),
        })
    directory = {"schema": PROVIDERS_SCHEMA, "providers": providers}
    ratings = None
    for doc in docs:
        if doc.get("schema") == RATINGS_SCHEMA and isinstance(doc.get("ratings"), list):
            rows = []
            for raw in doc["ratings"]:
                if not is_record(raw):
                    continue
                model = string_of(raw.get("model"))
                provider = string_of(raw.get("provider"))
                if model == "" or provider == "":
                    continue
                rows.append({
                    "model": model,
                    "provider": provider,
                    "rating_milli": number_or_none(raw.get("rating_milli")) or 0,
                    "context_length": number_or_none(raw.get("context_length")),
                    "tools": raw.get("tools") is True,
                    "vision": raw.get("vision") is True,
                    "json": raw.get("json") is True,
                    "free": raw.get("free") is True,
                })
            best = []
            if isinstance(doc.get("best"), list):
                for raw in doc["best"]:
                    if is_record(raw) and string_of(raw.get("id")) != "":
                        best.append({"id": string_of(raw.get("id")),
                                     "rating_milli": number_or_none(raw.get("rating_milli")) or 0})
            ratings = {"schema": RATINGS_SCHEMA, "ratings": rows, "best": best}
            break
    workload = None
    for doc in docs:
        if doc.get("schema") == WORKLOAD_SCHEMA and isinstance(doc.get("manifest"), list):
            manifest = []
            for raw in doc["manifest"]:
                if not is_record(raw):
                    continue
                request_id = string_of(raw.get("id"))
                if request_id == "":
                    continue
                requires = raw.get("requires") if is_record(raw.get("requires")) else {}
                entry = {
                    "id": request_id,
                    "api": "anthropic" if raw.get("api") == "anthropic" else "openai",
                    "stream": raw.get("stream") is True,
                    "weight": 2 if raw.get("weight") == 2 else 1,
                    "requires": {
                        "tools": requires.get("tools") is True,
                        "vision": requires.get("vision") is True,
                        "json": requires.get("json") is True,
                    },
                }
                if isinstance(requires.get("min_context"), (int, float)):
                    entry["requires"]["min_context"] = requires["min_context"]
                manifest.append(entry)
            workload = {"schema": WORKLOAD_SCHEMA, "manifest": manifest}
            break
    return directory, ratings, workload, None


def rating_of_model(ratings, provider_id, model_id):
    if ratings is None:
        return None
    for row in ratings["ratings"]:
        if row["provider"] == provider_id and row["model"] == model_id:
            return row
    return None


def model_free(model, row):
    """The quality table's free flag where it has a row, else the listing's."""
    if row is not None:
        return row["free"]
    return model.get("free") is True


def capabilities_of(model, row):
    if row is not None:
        return {
            "context_length": row["context_length"] if row.get("context_length") is not None else model.get("context_length"),
            "tools": row["tools"],
            "vision": row["vision"],
            "json": row["json"],
        }
    return {
        "context_length": model.get("context_length"),
        "tools": model.get("tools"),
        "vision": model.get("vision"),
        "json": model.get("json"),
    }


def meets_requirements(requires, caps):
    if requires.get("tools") is True and caps.get("tools") is not True:
        return False
    if requires.get("vision") is True and caps.get("vision") is not True:
        return False
    if requires.get("json") is True and caps.get("json") is not True:
        return False
    if requires.get("min_context") is not None:
        context = caps.get("context_length")
        if context is None or context < requires["min_context"]:
            return False
    return True


# ---------------------------------------------------------------------------
# HTTP helpers (standard library only; no proxy: the sandbox reaches the env
# directly, and http.client never consults one anyway).
# ---------------------------------------------------------------------------

def join_base(env_url, base):
    base = base.strip()
    if base.lower().startswith("http://") or base.lower().startswith("https://"):
        return base.rstrip("/")
    prefix = env_url.rstrip("/")
    if base == "":
        return prefix
    return prefix + ("" if base.startswith("/") else "/") + base.rstrip("/")


def auth_headers(auth, key):
    if key == "":
        return {}
    if auth == "bearer":
        return {"authorization": "Bearer " + key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def http_request(url, method, headers, body_bytes, timeout):
    """One HTTP request; returns (status, body bytes) or (0, b'') on failure."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "http":
        return 0, b""
    host = parsed.hostname or ""
    port = parsed.port or 80
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query
    connection = HTTPConnection(host, port, timeout=timeout)
    try:
        request_headers = dict(headers)
        if body_bytes is not None:
            request_headers["content-length"] = str(len(body_bytes))
        connection.request(method, path, body=body_bytes, headers=request_headers)
        response = connection.getresponse()
        data = response.read()
        return response.status, data
    except Exception:
        return 0, b""
    finally:
        try:
            connection.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# The catalog probe: each provider's catalog route, in its own wire.
# ---------------------------------------------------------------------------

def _model_in(wire, raw):
    if not is_record(raw):
        return None
    if wire == "gemini":
        model_id = string_of(raw.get("name")).replace("models/", "", 1)
    else:
        model_id = string_of(raw.get("id"))
    if model_id == "":
        return None
    context = number_or_none(raw.get("context_length"))
    if context is None:
        context = number_or_none(raw.get("context_window"))
    if context is None:
        context = number_or_none(raw.get("inputTokenLimit"))
    return {
        "id": model_id,
        "display_name": string_of(raw.get("display_name")) or string_of(raw.get("displayName")) or model_id,
        "free": bool_or_none(raw.get("free")),
        "context_length": context,
        "tools": bool_or_none(raw.get("tools")),
        "vision": bool_or_none(raw.get("vision")),
        "json": bool_or_none(raw.get("json")),
    }


def _models_in(wire, body):
    if not is_record(body):
        return []
    listing = body.get("models") if wire == "gemini" else body.get("data")
    if not isinstance(listing, list):
        return []
    out = []
    for raw in listing:
        model = _model_in(wire, raw)
        if model is not None:
            out.append(model)
    return out


def probe_provider(env_url, provider, timeout=PROBE_TIMEOUT_S):
    """Probe one provider's catalog route; a failure leaves it out."""
    base = join_base(env_url, provider["base"])
    url = base + ("/v1beta/models" if provider["wire"] == "gemini" else "/v1/models")
    headers = {"accept": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    status, data = http_request(url, "GET", headers, None, timeout)
    if status < 200 or status >= 300:
        return {"provider": provider, "ok": False, "status": status, "models": [],
                "error": "GET %s answered HTTP %s" % (url, status)}
    try:
        body = json.loads(data.decode("utf-8"))
    except Exception as exc:
        return {"provider": provider, "ok": False, "status": status, "models": [],
                "error": "GET %s answered non-JSON: %s" % (url, exc)}
    return {"provider": provider, "ok": True, "status": status,
            "models": _models_in(provider["wire"], body), "error": None}


def build_catalog(probes, ratings):
    """The catalog the probes and the quality table make: every provider that
    answered, every model it listed. Its base_url is the directory's own path
    (/p/<id>), never joined to the env URL: the env's origin differs between
    a live pass and its replay, and the catalog is part of the state export
    a replay must reproduce byte for byte."""
    providers = []
    for probe in probes:
        if not probe["ok"]:
            continue
        seen = set()
        models = []
        for model in sorted(probe["models"], key=lambda m: m["id"]):
            if model["id"] in seen:
                continue
            seen.add(model["id"])
            row = rating_of_model(ratings, probe["provider"]["id"], model["id"])
            context = None
            if row is not None and row.get("context_length") is not None:
                context = row["context_length"]
            elif model.get("context_length") is not None:
                context = model["context_length"]
            models.append({"id": model["id"], "free": model_free(model, row),
                           "context_length": context})
        providers.append({
            "id": probe["provider"]["id"],
            "base_url": probe["provider"]["base"],
            "key_env": probe["provider"]["key_env"],
            "wire": probe["provider"]["wire"],
            "models": models,
        })
    return {"providers": providers}


def served_models(probes, ratings):
    """The free models across the providers that answered, in the brief's
    provider order, models sorted by id: what GET /v1/models lists."""
    out = []
    for probe in probes:
        if not probe["ok"]:
            continue
        for model in sorted(probe["models"], key=lambda m: m["id"]):
            row = rating_of_model(ratings, probe["provider"]["id"], model["id"])
            if not model_free(model, row):
                continue
            caps = capabilities_of(model, row)
            out.append({
                "id": model["id"],
                "provider": probe["provider"]["id"],
                "display_name": model["display_name"],
                "context_length": caps.get("context_length"),
                "tools": caps.get("tools") is True,
                "vision": caps.get("vision") is True,
                "json": caps.get("json") is True,
            })
    return out


# ---------------------------------------------------------------------------
# The client's request, in one canonical shape.
#
# A message's content is kept as the client wrote it -- a string stays a
# string, a parts array stays a parts array -- so the provider's echo of it
# is the client's own. Images are normalised to {kind, url} pairs: a data URL
# carries its base64 inline, a plain URL is passed as a file reference.
# ---------------------------------------------------------------------------

def _blocks_text(content):
    """The text of a string or of a block array's text blocks, joined with \n."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if is_record(block) and block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts)


def _image_parts(content):
    """The image parts of a message's content, as {url} dicts, in order."""
    out = []
    if not isinstance(content, list):
        return out
    for block in content:
        if not is_record(block):
            continue
        if block.get("type") == "image_url" and is_record(block.get("image_url")):
            url = string_of(block["image_url"].get("url"))
            if url != "":
                out.append({"url": url})
        elif block.get("type") == "image" and is_record(block.get("source")):
            source = block["source"]
            if source.get("type") == "base64" and isinstance(source.get("data"), str):
                media = string_of(source.get("media_type")) or "image/png"
                out.append({"url": "data:%s;base64,%s" % (media, source["data"])})
            elif source.get("type") == "url" and isinstance(source.get("url"), str):
                out.append({"url": source["url"]})
    return out


def _data_url_parts(url):
    """A data URL as (mime_type, base64) or a plain URL as (None, url)."""
    if url.startswith("data:"):
        head, _, data = url.partition(",")
        mime = head[5:].split(";")[0] or "image/png"
        return mime, data
    return None, url


def extract_request(client_api, body):
    """The canonical request: system text, messages, tools, caps, stream."""
    canon = {
        "system": "",
        "messages": [],
        "tools": [],
        "tool_choice": None,
        "max_tokens": None,
        "temperature": None,
        "top_p": None,
        "stops": [],
        "json_mode": False,
        "stream": body.get("stream") is True,
    }
    if client_api == "openai":
        system_parts = []
        for message in body.get("messages") if isinstance(body.get("messages"), list) else []:
            if not is_record(message):
                continue
            role = message.get("role")
            if role == "system":
                text = _blocks_text(message.get("content"))
                if text != "":
                    system_parts.append(text)
        canon["system"] = "\n".join(system_parts)
        for message in body.get("messages") if isinstance(body.get("messages"), list) else []:
            if not is_record(message):
                continue
            role = message.get("role")
            if role not in ("user", "assistant", "tool"):
                continue
            entry = {"role": role, "content": message.get("content"),
                     "text": _blocks_text(message.get("content")),
                     "images": _image_parts(message.get("content"))}
            if role == "assistant":
                calls = []
                for call in message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []:
                    if is_record(call) and is_record(call.get("function")):
                        calls.append({
                            "id": string_of(call.get("id")),
                            "name": string_of(call["function"].get("name")),
                            "arguments": call["function"].get("arguments"),
                        })
                entry["tool_calls"] = calls
            if role == "tool":
                entry["tool_call_id"] = string_of(message.get("tool_call_id"))
            canon["messages"].append(entry)
        for tool in body.get("tools") if isinstance(body.get("tools"), list) else []:
            if is_record(tool) and is_record(tool.get("function")) and isinstance(tool["function"].get("name"), str):
                canon["tools"].append({
                    "name": tool["function"]["name"],
                    "description": string_of(tool["function"].get("description")),
                    "parameters": tool["function"].get("parameters")
                    if is_record(tool["function"].get("parameters")) else {"type": "object", "properties": {}},
                })
        if isinstance(body.get("tool_choice"), str):
            canon["tool_choice"] = body["tool_choice"]
        elif is_record(body.get("tool_choice")):
            canon["tool_choice"] = body["tool_choice"]
        response_format = body.get("response_format") if is_record(body.get("response_format")) else {}
        canon["json_mode"] = response_format.get("type") in ("json_object", "json_schema")
        canon["max_tokens"] = number_or_none(body.get("max_tokens"))
        if canon["max_tokens"] is None:
            canon["max_tokens"] = number_or_none(body.get("max_completion_tokens"))
        canon["stops"] = string_list(body.get("stop"))
    else:
        canon["system"] = _blocks_text(body.get("system"))
        for message in body.get("messages") if isinstance(body.get("messages"), list) else []:
            if not is_record(message):
                continue
            role = message.get("role")
            if role not in ("user", "assistant"):
                continue
            entry = {"role": role, "content": message.get("content"),
                     "text": _blocks_text(message.get("content")),
                     "images": _image_parts(message.get("content"))}
            if role == "assistant":
                calls = []
                if isinstance(message.get("content"), list):
                    for block in message["content"]:
                        if is_record(block) and block.get("type") == "tool_use":
                            calls.append({
                                "id": string_of(block.get("id")),
                                "name": string_of(block.get("name")),
                                "arguments": block.get("input"),
                            })
                entry["tool_calls"] = calls
            if role == "user" and isinstance(message.get("content"), list):
                results = []
                for block in message["content"]:
                    if is_record(block) and block.get("type") == "tool_result":
                        content = block.get("content")
                        if isinstance(content, str):
                            text = content
                        elif isinstance(content, list):
                            text = _blocks_text(content)
                            if text == "":
                                text = json.dumps(content)
                        else:
                            text = "" if content is None else json.dumps(content)
                        results.append({"tool_use_id": string_of(block.get("tool_use_id")),
                                        "text": text})
                entry["tool_results"] = results
            canon["messages"].append(entry)
        for tool in body.get("tools") if isinstance(body.get("tools"), list) else []:
            if is_record(tool) and isinstance(tool.get("name"), str):
                canon["tools"].append({
                    "name": tool["name"],
                    "description": string_of(tool.get("description")),
                    "parameters": tool.get("input_schema")
                    if is_record(tool.get("input_schema")) else {"type": "object", "properties": {}},
                })
        if is_record(body.get("tool_choice")):
            canon["tool_choice"] = body["tool_choice"]
        canon["max_tokens"] = number_or_none(body.get("max_tokens"))
        canon["stops"] = string_list(body.get("stop_sequences"))
    if isinstance(body.get("temperature"), (int, float)):
        canon["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        canon["top_p"] = body["top_p"]
    return canon


def requirements_from_body(client_api, body):
    """Requirements read off a request body, when the manifest has no row."""
    vision = False
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    for message in messages:
        if not is_record(message) or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if is_record(block) and block.get("type") in ("image", "image_url"):
                vision = True
    response_format = body.get("response_format") if is_record(body.get("response_format")) else {}
    return {
        "tools": isinstance(body.get("tools"), list) and len(body["tools"]) > 0,
        "vision": vision,
        "json": response_format.get("type") in ("json_object", "json_schema"),
    }


# ---------------------------------------------------------------------------
# The canonical request in each provider wire.
# ---------------------------------------------------------------------------

def build_openai_body(canon, model):
    messages = []
    if canon["system"] != "":
        messages.append({"role": "system", "content": canon["system"]})
    for entry in canon["messages"]:
        role = entry["role"]
        if role == "user":
            if isinstance(entry["content"], str):
                messages.append({"role": "user", "content": entry["content"]})
            elif isinstance(entry["content"], list):
                parts = []
                for block in entry["content"]:
                    if not is_record(block):
                        continue
                    if block.get("type") == "text" and isinstance(block.get("text"), str):
                        parts.append({"type": "text", "text": block["text"]})
                    elif block.get("type") == "image_url" and is_record(block.get("image_url")):
                        parts.append({"type": "image_url", "image_url": {"url": string_of(block["image_url"].get("url"))}})
                    elif block.get("type") == "image" and is_record(block.get("source")):
                        source = block["source"]
                        if source.get("type") == "base64" and isinstance(source.get("data"), str):
                            media = string_of(source.get("media_type")) or "image/png"
                            parts.append({"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (media, source["data"])}})
                        elif source.get("type") == "url" and isinstance(source.get("url"), str):
                            parts.append({"type": "image_url", "image_url": {"url": source["url"]}})
                if entry.get("tool_results"):
                    for result in entry["tool_results"]:
                        messages.append({"role": "tool", "tool_call_id": result["tool_use_id"],
                                         "content": result["text"]})
                if parts:
                    messages.append({"role": "user", "content": parts})
            else:
                messages.append({"role": "user", "content": entry["text"]})
        elif role == "assistant":
            message = {"role": "assistant", "content": entry["text"] if entry["text"] != "" else None}
            if entry.get("tool_calls"):
                message["tool_calls"] = [{
                    "id": call["id"], "type": "function",
                    "function": {"name": call["name"],
                                 "arguments": json.dumps(call["arguments"])
                                 if is_record(call["arguments"]) else string_of(call["arguments"])},
                } for call in entry["tool_calls"]]
            messages.append(message)
        elif role == "tool":
            messages.append({"role": "tool", "tool_call_id": entry.get("tool_call_id", ""),
                             "content": entry["text"]})
    out = {"model": model, "messages": messages}
    if canon["max_tokens"] is not None:
        out["max_tokens"] = canon["max_tokens"]
    if canon["temperature"] is not None:
        out["temperature"] = canon["temperature"]
    if canon["top_p"] is not None:
        out["top_p"] = canon["top_p"]
    if canon["stops"]:
        out["stop"] = canon["stops"]
    if canon["tools"]:
        out["tools"] = [{
            "type": "function",
            "function": {"name": tool["name"], "description": tool["description"],
                         "parameters": tool["parameters"]},
        } for tool in canon["tools"]]
    if isinstance(canon["tool_choice"], str):
        out["tool_choice"] = canon["tool_choice"]
    elif is_record(canon["tool_choice"]):
        choice = canon["tool_choice"]
        if choice.get("type") == "auto":
            out["tool_choice"] = "auto"
        elif choice.get("type") == "any":
            out["tool_choice"] = "required"
        elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
            out["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    if canon["json_mode"]:
        out["response_format"] = {"type": "json_object"}
    if canon["stream"]:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def build_anthropic_body(canon, model):
    messages = []
    for entry in canon["messages"]:
        role = entry["role"]
        if role == "user":
            if isinstance(entry["content"], str):
                blocks = [{"type": "text", "text": entry["content"]}]
            elif isinstance(entry["content"], list):
                blocks = []
                for block in entry["content"]:
                    if not is_record(block):
                        continue
                    if block.get("type") == "text" and isinstance(block.get("text"), str):
                        blocks.append({"type": "text", "text": block["text"]})
                    elif block.get("type") == "image_url" and is_record(block.get("image_url")):
                        url = string_of(block["image_url"].get("url"))
                        mime, data = _data_url_parts(url)
                        if mime is not None:
                            blocks.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}})
                        else:
                            blocks.append({"type": "image", "source": {"type": "url", "url": url}})
                    elif block.get("type") == "image" and is_record(block.get("source")):
                        blocks.append(block)
                for result in entry.get("tool_results") or []:
                    blocks.append({"type": "tool_result", "tool_use_id": result["tool_use_id"],
                                   "content": result["text"]})
            else:
                blocks = [{"type": "text", "text": entry["text"]}]
            if blocks:
                messages.append({"role": "user", "content": blocks})
        elif role == "assistant":
            blocks = []
            if entry["text"] != "":
                blocks.append({"type": "text", "text": entry["text"]})
            for call in entry.get("tool_calls") or []:
                arguments = call["arguments"]
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except Exception:
                        arguments = {}
                if not is_record(arguments):
                    arguments = {}
                blocks.append({"type": "tool_use", "id": call["id"] or "toolu_unknown",
                               "name": call["name"], "input": arguments})
            if blocks:
                messages.append({"role": "assistant", "content": blocks})
        elif role == "tool":
            messages.append({"role": "user", "content": [{
                "type": "tool_result", "tool_use_id": entry.get("tool_call_id", ""),
                "content": entry["text"]}]})
    out = {"model": model, "messages": messages}
    if canon["system"] != "":
        out["system"] = canon["system"]
    if canon["max_tokens"] is not None:
        out["max_tokens"] = canon["max_tokens"]
    if canon["temperature"] is not None:
        out["temperature"] = canon["temperature"]
    if canon["top_p"] is not None:
        out["top_p"] = canon["top_p"]
    if canon["stops"]:
        out["stop_sequences"] = canon["stops"]
    if canon["tools"]:
        out["tools"] = [{
            "name": tool["name"], "description": tool["description"],
            "input_schema": tool["parameters"],
        } for tool in canon["tools"]]
    if is_record(canon["tool_choice"]):
        out["tool_choice"] = canon["tool_choice"]
    elif isinstance(canon["tool_choice"], str):
        out["tool_choice"] = {"type": canon["tool_choice"]}
    if canon["stream"]:
        out["stream"] = True
    return out


def build_gemini_body(canon):
    contents = []
    for entry in canon["messages"]:
        if entry["role"] == "tool":
            continue
        parts = []
        if isinstance(entry["content"], str):
            if entry["content"] != "":
                parts.append({"text": entry["content"]})
        elif isinstance(entry["content"], list):
            for block in entry["content"]:
                if not is_record(block):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append({"text": block["text"]})
                elif block.get("type") == "image_url" and is_record(block.get("image_url")):
                    url = string_of(block["image_url"].get("url"))
                    mime, data = _data_url_parts(url)
                    if mime is not None:
                        parts.append({"inlineData": {"mimeType": mime, "data": data}})
                    else:
                        parts.append({"fileData": {"fileUri": url}})
                elif block.get("type") == "image" and is_record(block.get("source")):
                    source = block["source"]
                    if source.get("type") == "base64" and isinstance(source.get("data"), str):
                        parts.append({"inlineData": {"mimeType": string_of(source.get("media_type")) or "image/png",
                                                     "data": source["data"]}})
                    elif source.get("type") == "url" and isinstance(source.get("url"), str):
                        parts.append({"fileData": {"fileUri": source["url"]}})
        else:
            if entry["text"] != "":
                parts.append({"text": entry["text"]})
        if entry["role"] == "assistant":
            for call in entry.get("tool_calls") or []:
                parts.append({"functionCall": {"name": call["name"],
                                               "args": call["arguments"] if is_record(call["arguments"]) else {}}})
        if not parts:
            continue
        contents.append({"role": "model" if entry["role"] == "assistant" else "user",
                         "parts": parts})
    out = {"contents": contents}
    if canon["system"] != "":
        out["systemInstruction"] = {"parts": [{"text": canon["system"]}]}
    if canon["tools"]:
        out["tools"] = [{"functionDeclarations": [{
            "name": tool["name"], "description": tool["description"],
            "parameters": tool["parameters"],
        } for tool in canon["tools"]]}]
    config = {}
    if canon["max_tokens"] is not None:
        config["maxOutputTokens"] = canon["max_tokens"]
    if canon["stops"]:
        config["stopSequences"] = canon["stops"]
    if canon["json_mode"]:
        config["responseMimeType"] = "application/json"
    if canon["temperature"] is not None:
        config["temperature"] = canon["temperature"]
    if canon["top_p"] is not None:
        config["topP"] = canon["top_p"]
    if config:
        out["generationConfig"] = config
    return out


# ---------------------------------------------------------------------------
# One upstream call, and one provider answer parsed to the canonical shape:
# {text, input_tokens, output_tokens, stop_family}.
# ---------------------------------------------------------------------------

def call_provider(env_url, provider, canon, model, stream):
    """Call one provider; returns (status, raw body bytes)."""
    base = join_base(env_url, provider["base"])
    wire = provider["wire"]
    if wire == "openai":
        url = base + "/v1/chat/completions"
        body = build_openai_body(canon, model)
    elif wire == "anthropic":
        url = base + "/v1/messages"
        body = build_anthropic_body(canon, model)
    else:
        action = "streamGenerateContent" if stream else "generateContent"
        url = base + "/v1beta/models/" + urllib.parse.quote(model, safe="") + ":" + action
        body = build_gemini_body(canon)
    headers = {"content-type": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if wire == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    if stream:
        headers["accept"] = "text/event-stream"
    return http_request(url, "POST", headers,
                        json.dumps(body).encode("utf-8"), PROVIDER_TIMEOUT_S)


def _sse_frames(raw):
    """SSE body bytes as [(event, data)] frames, in order."""
    text = raw.decode("utf-8", errors="replace").replace("\r\n", "\n")
    frames = []
    for frame in text.split("\n\n"):
        if frame.strip() == "":
            continue
        event = None
        data = []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data.append(line[len("data:"):].lstrip(" "))
        if data:
            frames.append((event, "\n".join(data)))
    return frames


def _text_of_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(block["text"] for block in content
                         if is_record(block) and block.get("type") == "text"
                         and isinstance(block.get("text"), str))
    return ""


def parse_openai_answer(raw, stream):
    try:
        if not stream:
            value = json.loads(raw.decode("utf-8"))
            if not is_record(value):
                return None
            choices = value.get("choices") if isinstance(value.get("choices"), list) else []
            if not choices or not is_record(choices[0]):
                return None
            choice = choices[0]
            message = choice.get("message") if is_record(choice.get("message")) else {}
            text = _text_of_content(message.get("content"))
            usage = value.get("usage") if is_record(value.get("usage")) else {}
            return {"text": text,
                    "input_tokens": number_or_none(usage.get("prompt_tokens")) or 0,
                    "output_tokens": number_or_none(usage.get("completion_tokens")) or 0,
                    "stop_family": STOP_FAMILY.get(choice.get("finish_reason"), 0)}
        text = ""
        input_tokens = 0
        output_tokens = 0
        stop_family = 0
        for _event, data in _sse_frames(raw):
            if data.strip() == "[DONE]":
                continue
            try:
                chunk = json.loads(data)
            except Exception:
                continue
            if not is_record(chunk):
                continue
            if is_record(chunk.get("usage")):
                input_tokens = number_or_none(chunk["usage"].get("prompt_tokens")) or 0
                output_tokens = number_or_none(chunk["usage"].get("completion_tokens")) or 0
            choices = chunk.get("choices") if isinstance(chunk.get("choices"), list) else []
            if not choices or not is_record(choices[0]):
                continue
            choice = choices[0]
            delta = choice.get("delta") if is_record(choice.get("delta")) else {}
            if isinstance(delta.get("content"), str):
                text += delta["content"]
            if isinstance(choice.get("finish_reason"), str):
                stop_family = STOP_FAMILY.get(choice["finish_reason"], 0)
        return {"text": text, "input_tokens": input_tokens, "output_tokens": output_tokens,
                "stop_family": stop_family}
    except Exception:
        return None


def parse_anthropic_answer(raw, stream):
    try:
        if not stream:
            value = json.loads(raw.decode("utf-8"))
            if not is_record(value) or value.get("type") != "message":
                return None
            usage = value.get("usage") if is_record(value.get("usage")) else {}
            return {"text": _text_of_content(value.get("content")),
                    "input_tokens": number_or_none(usage.get("input_tokens")) or 0,
                    "output_tokens": number_or_none(usage.get("output_tokens")) or 0,
                    "stop_family": STOP_FAMILY.get(value.get("stop_reason"), 0)}
        text = ""
        input_tokens = 0
        output_tokens = 0
        stop_family = 0
        for event, data in _sse_frames(raw):
            try:
                frame = json.loads(data)
            except Exception:
                continue
            if not is_record(frame):
                continue
            if event == "message_start" and is_record(frame.get("message")):
                usage = frame["message"].get("usage") if is_record(frame["message"].get("usage")) else {}
                input_tokens = number_or_none(usage.get("input_tokens")) or 0
            elif event == "content_block_delta" and is_record(frame.get("delta")):
                if frame["delta"].get("type") == "text_delta" and isinstance(frame["delta"].get("text"), str):
                    text += frame["delta"]["text"]
            elif event == "message_delta":
                if is_record(frame.get("delta")) and isinstance(frame["delta"].get("stop_reason"), str):
                    stop_family = STOP_FAMILY.get(frame["delta"]["stop_reason"], 0)
                if is_record(frame.get("usage")):
                    output_tokens = number_or_none(frame["usage"].get("output_tokens")) or 0
        return {"text": text, "input_tokens": input_tokens, "output_tokens": output_tokens,
                "stop_family": stop_family}
    except Exception:
        return None


def parse_gemini_answer(raw, stream):
    try:
        if not stream:
            value = json.loads(raw.decode("utf-8"))
            if not is_record(value):
                return None
            candidates = value.get("candidates") if isinstance(value.get("candidates"), list) else []
            if not candidates or not is_record(candidates[0]):
                return None
            candidate = candidates[0]
            content = candidate.get("content") if is_record(candidate.get("content")) else {}
            parts = content.get("parts") if isinstance(content.get("parts"), list) else []
            text = "".join(part["text"] for part in parts
                           if is_record(part) and isinstance(part.get("text"), str))
            usage = value.get("usageMetadata") if is_record(value.get("usageMetadata")) else {}
            return {"text": text,
                    "input_tokens": number_or_none(usage.get("promptTokenCount")) or 0,
                    "output_tokens": number_or_none(usage.get("candidatesTokenCount")) or 0,
                    "stop_family": STOP_FAMILY.get(candidate.get("finishReason"), 0)}
        text = ""
        input_tokens = 0
        output_tokens = 0
        stop_family = 0
        for _event, data in _sse_frames(raw):
            try:
                frame = json.loads(data)
            except Exception:
                continue
            if not is_record(frame):
                continue
            candidates = frame.get("candidates") if isinstance(frame.get("candidates"), list) else []
            if candidates and is_record(candidates[0]):
                candidate = candidates[0]
                content = candidate.get("content") if is_record(candidate.get("content")) else {}
                parts = content.get("parts") if isinstance(content.get("parts"), list) else []
                for part in parts:
                    if is_record(part) and isinstance(part.get("text"), str):
                        text += part["text"]
                if isinstance(candidate.get("finishReason"), str):
                    stop_family = STOP_FAMILY.get(candidate["finishReason"], 0)
            if is_record(frame.get("usageMetadata")):
                usage = frame["usageMetadata"]
                input_tokens = number_or_none(usage.get("promptTokenCount")) or 0
                output_tokens = number_or_none(usage.get("candidatesTokenCount")) or 0
        return {"text": text, "input_tokens": input_tokens, "output_tokens": output_tokens,
                "stop_family": stop_family}
    except Exception:
        return None


def parse_provider_answer(wire, raw, stream):
    if wire == "openai":
        return parse_openai_answer(raw, stream)
    if wire == "anthropic":
        return parse_anthropic_answer(raw, stream)
    return parse_gemini_answer(raw, stream)


# ---------------------------------------------------------------------------
# The canonical answer in the client's wire.
# ---------------------------------------------------------------------------

def _split_text(text):
    pieces = []
    for at in range(0, len(text), SSE_DELTA_CHARS):
        pieces.append(text[at:at + SSE_DELTA_CHARS])
    return pieces if pieces else [""]


def render_openai_json(answer, model):
    return {
        "id": "chatcmpl-sim",
        "object": "chat.completion",
        "created": FIXED_OPENAI_CREATED,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer["text"]},
            "finish_reason": OPENAI_STOP[answer["stop_family"]],
            "logprobs": None,
        }],
        "usage": {
            "prompt_tokens": answer["input_tokens"],
            "completion_tokens": answer["output_tokens"],
            "total_tokens": answer["input_tokens"] + answer["output_tokens"],
        },
    }


def render_openai_sse(answer, model):
    def chunk(delta, finish):
        return {"id": "chatcmpl-sim", "object": "chat.completion.chunk",
                "created": FIXED_OPENAI_CREATED, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    frames = ["data: %s\n\n" % json.dumps(chunk({"role": "assistant", "content": ""}, None))]
    for piece in _split_text(answer["text"]):
        frames.append("data: %s\n\n" % json.dumps(chunk({"content": piece}, None)))
    frames.append("data: %s\n\n" % json.dumps(chunk({}, OPENAI_STOP[answer["stop_family"]])))
    frames.append("data: %s\n\n" % json.dumps({
        "id": "chatcmpl-sim", "object": "chat.completion.chunk",
        "created": FIXED_OPENAI_CREATED, "model": model, "choices": [],
        "usage": {"prompt_tokens": answer["input_tokens"],
                  "completion_tokens": answer["output_tokens"],
                  "total_tokens": answer["input_tokens"] + answer["output_tokens"]}}))
    frames.append("data: [DONE]\n\n")
    return "".join(frames)


def render_anthropic_json(answer, model):
    return {
        "id": "msg_aggregator",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": answer["text"]}],
        "stop_reason": ANTHROPIC_STOP[answer["stop_family"]],
        "stop_sequence": None,
        "usage": {"input_tokens": answer["input_tokens"],
                  "output_tokens": answer["output_tokens"]},
    }


def _sse_event(event, data):
    return "event: %s\ndata: %s\n\n" % (event, json.dumps(data))


def render_anthropic_sse(answer, model):
    frames = [_sse_event("message_start", {
        "type": "message_start",
        "message": {"id": "msg_aggregator", "type": "message", "role": "assistant",
                    "model": model, "content": [], "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": answer["input_tokens"], "output_tokens": 0}}})]
    frames.append(_sse_event("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""}}))
    for piece in _split_text(answer["text"]):
        frames.append(_sse_event("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": piece}}))
    frames.append(_sse_event("content_block_stop", {"type": "content_block_stop", "index": 0}))
    frames.append(_sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": ANTHROPIC_STOP[answer["stop_family"]], "stop_sequence": None},
        "usage": {"output_tokens": answer["output_tokens"]}}))
    frames.append(_sse_event("message_stop", {"type": "message_stop"}))
    return "".join(frames)


# ---------------------------------------------------------------------------
# Routing: best free model first, fail over on a refusal.
# ---------------------------------------------------------------------------

def candidate_models(probes, ratings, requires, body_has_json_format):
    """Every (provider, model) that may serve the request, best rating first.

    A JSON request the client asked in JSON (an OpenAI body carrying
    response_format) can only go to a wire that carries it: OpenAI, or Gemini
    via responseMimeType. The Anthropic wire echoes json_mode false, so it
    cannot serve one. An Anthropic client's JSON request carries no
    response_format, so no wire is excluded for it."""
    candidates = []
    for probe in probes:
        if not probe["ok"]:
            continue
        provider = probe["provider"]
        for model in probe["models"]:
            row = rating_of_model(ratings, provider["id"], model["id"])
            if not model_free(model, row):
                continue
            if row is None:
                continue
            if not meets_requirements(requires, capabilities_of(model, row)):
                continue
            if body_has_json_format and provider["wire"] == "anthropic":
                continue
            candidates.append((row["rating_milli"], provider["id"], model["id"], provider, row))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    return candidates


def serve_request(state, client_api, body):
    """Serve one client request; returns (status, content_type, body_text)."""
    canon = extract_request(client_api, body)
    requires = state.requirements_for(client_api, body)
    body_has_json_format = client_api == "openai" and canon["json_mode"]
    candidates = candidate_models(state.probes, state.ratings, requires, body_has_json_format)
    stream = canon["stream"]
    for _rating, _provider_id, model_id, provider, _row in candidates:
        status, raw = call_provider(state.env_url, provider, canon, model_id, stream)
        if status < 200 or status >= 300:
            if status != 0:
                log("refused: %s/%s answered HTTP %s" % (provider["id"], model_id, status))
            else:
                log("unreachable: %s/%s" % (provider["id"], model_id))
            continue
        answer = parse_provider_answer(provider["wire"], raw, stream)
        if answer is None:
            # A 2xx is already a successful provider call: calling another
            # would leave two of them, so answer the failure, not a retry.
            log("unparseable answer from %s/%s" % (provider["id"], model_id))
            break
        if client_api == "openai":
            if stream:
                return 200, "text/event-stream", render_openai_sse(answer, model_id)
            return 200, "application/json", json.dumps(render_openai_json(answer, model_id))
        if stream:
            return 200, "text/event-stream", render_anthropic_sse(answer, model_id)
        return 200, "application/json", json.dumps(render_anthropic_json(answer, model_id))
    message = "no free model on the directory meets this request"
    if client_api == "anthropic":
        return 503, "application/json", json.dumps(
            {"type": "error", "error": {"type": "overloaded_error", "message": message}})
    return 503, "application/json", json.dumps(
        {"error": {"message": message, "type": "no_free_model", "code": None}})


# ---------------------------------------------------------------------------
# Server state: the brief, the probes, the catalog, the manifest cursor.
# ---------------------------------------------------------------------------

class State:
    def __init__(self, env_url, directory, ratings, workload):
        self.env_url = env_url
        self.directory = directory
        self.ratings = ratings
        self.workload = workload
        self.probes = []
        self.catalog = {"providers": []}
        self.models = []
        self.manifest_index = 0
        self.lock = threading.Lock()

    def load(self):
        if self.env_url != "":
            self.probes = [probe_provider(self.env_url, provider)
                           for provider in self.directory["providers"]]
        else:
            self.probes = [{"provider": provider, "ok": False, "status": 0,
                            "models": [], "error": "no env URL"}
                           for provider in self.directory["providers"]]
        for probe in self.probes:
            if not probe["ok"] and probe.get("error"):
                log(probe["error"])
        self.catalog = build_catalog(self.probes, self.ratings)
        self.models = served_models(self.probes, self.ratings)

    def requirements_for(self, client_api, body):
        """The manifest's row by arrival order, else the body's own."""
        manifest = self.workload["manifest"] if self.workload is not None else []
        with self.lock:
            index = self.manifest_index
            self.manifest_index += 1
        if index < len(manifest):
            entry = manifest[index]
            if entry["api"] == client_api and entry["stream"] == (body.get("stream") is True):
                return entry["requires"]
        return requirements_from_body(client_api, body)


def openai_model_list(models):
    return {
        "object": "list",
        "data": [{
            "id": model["id"],
            "object": "model",
            "created": FIXED_OPENAI_CREATED,
            "owned_by": model["provider"],
            "free": True,
            "context_length": model["context_length"],
            "tools": model["tools"],
            "vision": model["vision"],
            "json": model["json"],
        } for model in models],
    }


def anthropic_model_list(models):
    return {
        "data": [{
            "type": "model",
            "id": model["id"],
            "display_name": model["display_name"],
            "created_at": FIXED_ANTHROPIC_CREATED_AT,
            "free": True,
            "context_length": model["context_length"],
            "tools": model["tools"],
            "vision": model["vision"],
            "json": model["json"],
        } for model in models],
        "has_more": False,
    }


def openai_error(message, error_type, code):
    return {"error": {"message": message, "type": error_type, "code": code}}


def anthropic_error(error_type, message):
    return {"type": "error", "error": {"type": error_type, "message": message}}


class Handler(BaseHTTPRequestHandler):
    state = None
    protocol_version = "HTTP/1.1"

    def log_message(self, _format, *_args):
        pass

    def _send_json(self, status, payload):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_text(self, status, content_type, text):
        data = text.encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, status, error_type, message):
        if self.path.startswith("/v1/messages"):
            self._send_json(status, anthropic_error(error_type, message))
        else:
            self._send_json(status, openai_error(message, error_type, None))

    def _read_json(self):
        try:
            length = int(self.headers.get("content-length") or 0)
        except ValueError:
            return None
        if length <= 0 or length > MAX_BODY_BYTES:
            return None
        try:
            raw = self.rfile.read(length)
        except Exception:
            return None
        try:
            value = json.loads(raw.decode("utf-8"))
        except Exception:
            return None
        return value if is_record(value) else None

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/v1/models":
            if isinstance(self.headers.get("x-api-key"), str):
                self._send_json(200, anthropic_model_list(self.state.models))
            else:
                self._send_json(200, openai_model_list(self.state.models))
            return
        self._send_error(404, "not_found", "no route GET %s" % parsed.path)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/v1/chat/completions":
            client_api = "openai"
        elif parsed.path == "/v1/messages":
            client_api = "anthropic"
        else:
            self._send_error(404, "not_found", "no route POST %s" % parsed.path)
            return
        body = self._read_json()
        if body is None:
            self._send_error(400, "invalid_request_error",
                             "the request body is not one JSON object")
            return
        try:
            status, content_type, text = serve_request(self.state, client_api, body)
        except Exception as exc:
            log("serve failed: %s" % exc)
            self._send_error(500, "internal_error", "the server failed to answer")
            return
        if content_type == "text/event-stream":
            data = text.encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "text/event-stream")
            self.send_header("cache-control", "no-cache")
            self.send_header("connection", "keep-alive")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self._send_text(status, content_type, text)


# ---------------------------------------------------------------------------
# Register, poll, run.
# ---------------------------------------------------------------------------

def register(env_url, port, catalog):
    body = json.dumps({"protocol_version": PROTOCOL_VERSION, "port": port,
                       "catalog": catalog}, separators=(",", ":")).encode("utf-8")
    for attempt in range(5):
        status, raw = http_request(env_url + "/register", "POST",
                                   {"content-type": "application/json"}, body,
                                   REGISTER_TIMEOUT_S)
        if status == 200:
            try:
                reply = json.loads(raw.decode("utf-8"))
            except Exception:
                reply = {}
            if is_record(reply) and isinstance(reply.get("requests"), int):
                log("registered: the workload holds %s requests" % reply["requests"])
            return True
        log("register attempt %s answered HTTP %s" % (attempt + 1, status))
        time.sleep(1)
    return False


def poll_done(env_url, server, deadline):
    """Poll GET /work until it says done; then shut the server down."""
    while True:
        if time.monotonic() > deadline:
            log("the trial wall is nearly spent and /work never said done")
            return False
        status, raw = http_request(env_url + "/work", "GET", {}, None, WORK_TIMEOUT_S)
        if status == 200:
            try:
                work = json.loads(raw.decode("utf-8"))
            except Exception:
                work = None
            if is_record(work) and work.get("done") is True:
                return True
        time.sleep(POLL_INTERVAL_S)


def main():
    env_url = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
    prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE", "")
    directory, ratings, workload, error = read_brief(prompt_file)
    if error is not None:
        log(error)
    state = State(env_url, directory, ratings, workload)
    state.load()
    if "--list" in sys.argv:
        # The same document the register body carries, byte for byte.
        text = json.dumps(state.catalog, separators=(",", ":")) + "\n"
        sys.stdout.write(text)
        sys.stdout.flush()
        return 0
    if env_url == "":
        log("MAKEFUN_ENV_URL is not set")
        return 1
    Handler.state = state
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    server.daemon_threads = True
    port = server.server_address[1]
    if port == 8080:
        log("the OS picked 8080, which the env owns on one layout: restart the trial")
        return 1
    log("listening on 0.0.0.0:%s with %s answered providers and %s free models"
        % (port, len(state.catalog["providers"]), len(state.models)))
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    if not register(env_url, port, state.catalog):
        log("POST %s/register failed" % env_url)
        server.shutdown()
        return 1
    deadline = time.monotonic() + 170
    done = poll_done(env_url, server, deadline)
    server.shutdown()
    server.server_close()
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
