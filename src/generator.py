import json
import re
import threading
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
from llm_sdk import Small_LLM_Model


_BYTE_DECODER: dict[str, int] | None = None


def _build_byte_decoder() -> dict[str, int]:
    """
    Inverse of GPT-2/Qwen's byte-to-unicode map used
      by tokenizer.json vocab entries.

    Byte-level BPE stores each of the 256 raw byte values as one printable
    unicode character so the vocab is safe to embed in JSON (e.g. a literal
    space is stored as "Ġ", a newline as "Ċ"). This rebuilds the mapping from
    unicode character back to the byte it represents.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def decode_token(token: str) -> str:
    """
    Convert a raw vocab entry (e.g. "Ġhello", "Ċ") into the text it represents.

    Special tokens (e.g. "<|im_end|>") aren't byte-mapped, so any character
    missing from the byte decoder is passed through as UTF-8 unchanged.
    """
    global _BYTE_DECODER
    if _BYTE_DECODER is None:
        _BYTE_DECODER = _build_byte_decoder()

    raw_bytes = bytearray()
    for ch in token:
        if ch in _BYTE_DECODER:
            raw_bytes.append(_BYTE_DECODER[ch])
        else:
            raw_bytes.extend(ch.encode("utf-8"))
    return raw_bytes.decode("utf-8", errors="ignore")


IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
# Qwen3's way to switch thinking off: a pre-closed, empty block.
EMPTY_THINK = "<think>\n\n</think>\n\n"

MAX_STRING_TOKENS = 64
MAX_NUMBER_TOKENS = 24
_SPECIAL_RE = re.compile(r"^<\|.*\|>$|^</?(think|tool_call|tool_response)>$")
_NUMBER_PREFIX_RE = re.compile(r"^-?\d*\.?\d*$")
_NUMBER_RE = re.compile(r"^-?\d+(\.\d+)?$")

# Token tables are built once per vocab and shared by all worker threads.
_TABLES: dict[int, dict[str, Any]] = {}
_TABLES_LOCK = threading.Lock()


def build_prompt(prompt: str, functions: list[dict[str, Any]]) -> str:
    """
    Short chat-formatted prompt: every prompt token is re-processed on each
    decoding step, so a compact function list directly saves time.
    """
    lines = []
    for fn in functions:
        params = ", ".join(
            f"{name}: {spec.get('type', 'string')}"
            for name, spec in fn.get("parameters", {}).items()
        )
        lines.append(
            f"- {fn['name']}({params}): {fn.get('description', '')}"
        )
    fns = "\n".join(lines)
    return (
        f"{IM_START}system\n"
        f"You are a function calling assistant. Pick the function that "
        f"answers the user request and extract its arguments.\n"
        f"Functions:\n{fns}\n"
        f'Answer as JSON: {{"name": ..., "parameters": {{...}}}}{IM_END}\n'
        f"{IM_START}user\n{prompt}{IM_END}\n"
        f"{IM_START}assistant\n{EMPTY_THINK}"
    )


def _get_tables(vocab: dict[int, str]) -> dict[str, Any]:
    """
    Decode every vocab entry once and precompute boolean masks of which
    tokens are allowed in each JSON position (string body, number, ...).
    """
    key = id(vocab)
    with _TABLES_LOCK:
        if key in _TABLES:
            return _TABLES[key]
        size = max(vocab) + 1
        texts = [""] * size
        special = np.zeros(size, dtype=bool)
        for tid, raw in vocab.items():
            texts[tid] = decode_token(raw)
            special[tid] = bool(_SPECIAL_RE.match(raw))
        nonempty = np.array([t != "" for t in texts]) & ~special
        string_body = nonempty & np.array(
            ['"' not in t and "\n" not in t for t in texts]
        )
        string_end = nonempty & np.array([t.startswith('"') for t in texts])
        number_body = nonempty & np.array(
            [all(c in "0123456789.-" for c in t) for t in texts]
        )
        number_end = nonempty & np.array(
            [t.lstrip().startswith((",", "}")) for t in texts]
        )
        text_to_ids: dict[str, list[int]] = {}
        for tid, text in enumerate(texts):
            if nonempty[tid]:
                text_to_ids.setdefault(text, []).append(tid)
        table = {
            "texts": texts,
            "string_body": string_body,
            "string_end": string_end,
            "number_body": number_body,
            "number_end": number_end,
            "text_to_ids": text_to_ids,
        }
        _TABLES[key] = table
        return table


def _pad(mask: npt.NDArray[np.bool_], size: int) -> npt.NDArray[np.bool_]:
    """The model's logit vector can be longer than the vocab file."""
    if len(mask) >= size:
        return mask[:size]
    return np.concatenate([mask, np.zeros(size - len(mask), dtype=bool)])


def _pick(
    logits: npt.NDArray[np.float32],
    mask: npt.NDArray[np.bool_],
    texts: list[str],
    accept: Callable[[str], bool] | None = None,
) -> int:
    """
    Constrained greedy choice: best-scoring token among those allowed by
    `mask` (others are treated as -inf) that also passes `accept`.
    """
    allowed = np.flatnonzero(_pad(mask, len(logits)))
    if len(allowed) == 0:
        raise ValueError("No valid token for current schema position")
    if accept is None:
        return int(allowed[np.argmax(logits[allowed])])
    for tid in allowed[np.argsort(-logits[allowed])]:
        if accept(texts[tid]):
            return int(tid)
    raise ValueError("No valid token for current schema position")


def _force(model: Small_LLM_Model, ids: list[int], text: str) -> None:
    """Append fixed JSON structure without asking the model."""
    ids.extend(model.encode(text)[0].tolist())


def _logits(model: Small_LLM_Model, ids: list[int]) -> npt.NDArray[np.float32]:
    return np.asarray(model.get_logits_from_input_ids(ids), dtype=np.float32)


def _common_prefix(words: list[str]) -> str:
    prefix = words[0]
    for w in words[1:]:
        while not w.startswith(prefix):
            prefix = prefix[:-1]
    return prefix


def _choose_name(
    model: Small_LLM_Model,
    ids: list[int],
    names: list[str],
    table: dict[str, Any],
) -> str:
    """
    Let the model pick the function name, restricted to tokens that keep
    the text a prefix of some real function name. Shared prefixes (e.g.
    "fn_") are forced, so the model is only queried where names diverge.
    """
    text_to_ids = table["text_to_ids"]
    texts = table["texts"]
    chosen = ""
    while True:
        remaining = [n for n in names if n.startswith(chosen)]
        if chosen in remaining and len(remaining) == 1:
            return chosen
        common = _common_prefix(remaining)
        if len(common) > len(chosen):
            _force(model, ids, common[len(chosen):])
            chosen = common
            continue
        mask = np.zeros(len(texts), dtype=bool)
        for name in remaining:
            rest = name[len(chosen):]
            for k in range(1, len(rest) + 1):
                for tid in text_to_ids.get(rest[:k], []):
                    mask[tid] = True
        if chosen in remaining:
            # Allow stopping here when a name is a prefix of another.
            mask |= table["string_end"]
        tid = _pick(_logits(model, ids), mask, texts)
        piece = texts[tid]
        if piece.startswith('"'):
            return chosen
        ids.append(tid)
        chosen += piece


def _gen_string(
    model: Small_LLM_Model, ids: list[int], table: dict[str, Any]
) -> str:
    """Generate a JSON string body; the opening quote is already forced."""
    texts = table["texts"]
    mask = table["string_body"] | table["string_end"]
    out = ""
    for _ in range(MAX_STRING_TOKENS):
        tid = _pick(_logits(model, ids), mask, texts)
        piece = texts[tid]
        escaped = (len(out) - len(out.rstrip("\\"))) % 2 == 1
        if piece.startswith('"') and not escaped:
            break
        if piece.startswith('"'):
            # \" is a quote inside the string, not its end.
            _force(model, ids, '"')
            out += '"'
            continue
        ids.append(tid)
        out += piece
    try:
        value: str = json.loads(f'"{out}"')
        return value
    except json.JSONDecodeError:
        return out


def _gen_number(
    model: Small_LLM_Model,
    ids: list[int],
    table: dict[str, Any],
    integer: bool,
) -> int | float:
    """Generate digits (and '.'/'-' for floats) until the model ends it."""
    texts = table["texts"]
    out = ""
    for _ in range(MAX_NUMBER_TOKENS):
        complete = bool(_NUMBER_RE.match(out))
        mask = table["number_body"]
        if complete:
            mask = mask | table["number_end"]

        def accept(piece: str, out: str = out) -> bool:
            if piece.lstrip().startswith((",", "}")):
                return True
            cand = out + piece
            if integer and "." in cand:
                return False
            return bool(_NUMBER_PREFIX_RE.match(cand))

        tid = _pick(_logits(model, ids), mask, texts, accept)
        piece = texts[tid]
        if piece.lstrip().startswith((",", "}")):
            break
        ids.append(tid)
        out += piece
    if not _NUMBER_RE.match(out):
        out = out.rstrip(".") or "0"
        if out in ("-", ""):
            out = "0"
    return int(float(out)) if integer else float(out)


def _gen_bool(model: Small_LLM_Model, ids: list[int]) -> bool:
    logits = _logits(model, ids)
    t_id = model.encode("true")[0].tolist()[0]
    f_id = model.encode("false")[0].tolist()[0]
    return bool(logits[t_id] >= logits[f_id])


def generate_function_call(
    model: Small_LLM_Model,
    prompt: str,
    functions: list[dict[str, Any]],
    vocab: dict[int, str],
) -> dict[str, Any]:
    """
    Runs constrained decoding to generate a valid function call JSON.
    Returns {"name": ..., "parameters": {...}}

    The JSON skeleton is forced token by token; the model is only queried
    to choose the function name and to fill in each parameter value, with
    invalid tokens masked out according to the parameter's type.
    """
    if not functions:
        raise ValueError("No functions defined")
    table = _get_tables(vocab)
    print(
        f"[GEN] Starting generation for: {prompt[:60]}"
        f"...\n by[TID]{threading.get_ident()}"
    )
    ids = model.encode(build_prompt(prompt, functions))[0].tolist()
    prompt_len = len(ids)

    _force(model, ids, '{"name": "')
    names = [fn["name"] for fn in functions]
    name = _choose_name(model, ids, names, table)
    fn = next(f for f in functions if f["name"] == name)

    params: dict[str, Any] = {}
    _force(model, ids, '", "parameters": {')
    for i, (key, spec) in enumerate(fn.get("parameters", {}).items()):
        sep = ", " if i else ""
        ptype = spec.get("type", "string")
        if ptype == "integer":
            _force(model, ids, f'{sep}"{key}": ')
            params[key] = _gen_number(model, ids, table,
                                      ptype == "integer")
        elif ptype in ("number", "boolean"):
            _force(model, ids, f'{sep}"{key}": ')
            params[key] = _gen_bool(model, ids)
            _force(model, ids, json.dumps(params[key]))
        else:
            _force(model, ids, f'{sep}"{key}": "')
            params[key] = _gen_string(model, ids, table)
            _force(model, ids, '"')

    print(
        f"[GEN] ✓ {name} ({len(ids) - prompt_len} tokens)"
        f" by[TID]{threading.get_ident()}"
    )
    return {"name": name, "parameters": params}
