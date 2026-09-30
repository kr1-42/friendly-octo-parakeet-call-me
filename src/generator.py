import json
import numpy as np
from llm_sdk import Small_LLM_Model
import threading


_BYTE_DECODER: dict[str, int] | None = None


def _build_byte_decoder() -> dict[str, int]:
    """
    Inverse of GPT-2/Qwen's byte-to-unicode map used by tokenizer.json vocab entries.

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


def build_prompt(prompt: str, functions: list[dict]) -> str:
    fns = json.dumps(functions, indent=2)
    return (
        f"You are a function calling assistant.\n"
        f"Available functions:\n{fns}\n\n"
        f"User request: {prompt}\n\n"
        f"Respond with ONLY a JSON object in this exact format:\n"
        f'{{"name": "<function_name>", "parameters": {{<args>}}}}\n'
    )


def parse_state(partial: str, functions: list[dict]) -> dict:
    """
    Returns a dict describing where we are in the JSON generation.
    e.g. {"phase": "name"} or {"phase": "param_value", "param": "a", "fn": "fn_add_numbers"}
    """
    # TODO: implement — for now just return the partial
    return {"partial": partial}


def is_valid_continuation(
    candidate: str, functions: list[dict], schema_state: dict
) -> bool:
    """
    Very permissive validator. Let the model generate freely.
    Only reject truly invalid patterns.
    """
    if not candidate:
        return True

    # Try to parse as complete JSON first
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict) and "name" in obj and "parameters" in obj:
            func_names = {fn["name"] for fn in functions}
            return obj["name"] in func_names
        return False
    except json.JSONDecodeError:
        pass

    # For incomplete JSON, almost always accept
    # Just reject the most obvious errors
    if not candidate.startswith("{"):
        return False

    # Basic bracket balance check
    if candidate.count("{") <= candidate.count("}"):
        return False
    if candidate.count("[") < candidate.count("]"):
        return False

    # Reject multiple root objects
    if "}{" in candidate or "} {" in candidate:
        return False

    # Accept everything else - let model generate
    return True


def get_valid_token_ids(
    partial: str, vocab: dict[int, str], functions: list[dict], schema_state: dict
) -> set[int]:
    """
    Given what we've generated so far, return the set of token IDs
    that are valid to generate next.

    For now, return all tokens - let the model generate freely and we'll
    catch valid JSON when it appears.
    """
    # Return all token IDs - no constraint
    return set(vocab.keys())


def generate_function_call(
    model: Small_LLM_Model,
    prompt: str,
    functions: list[dict],
    vocab: dict[int, str],
    max_tokens: int = 200,
) -> dict:
    """
    Runs constrained decoding to generate a valid function call JSON.
    Returns {"name": ..., "parameters": {...}}
    """
    # Build prompt that tells the model what functions exist
    system = build_prompt(prompt, functions)
    print(f"[GEN] Starting generation for: {prompt[:60]}...\n by[TID]{threading.get_ident()}")
    input_ids = model.encode(system)[0].tolist()  # flatten tensor to list

    generated_ids = []
    partial_json = ""
    result = None

    for step in range(max_tokens):
        logits = model.get_logits_from_input_ids(input_ids + generated_ids)
        logits = list(logits)

        next_id = int(np.argmax(logits))
        next_token = decode_token(vocab[next_id])

        generated_ids.append(next_id)
        partial_json += next_token

        if step % 20 == 0:
            print(f"[GEN] Step {step}: generated {len(partial_json)} chars")

        # Search for all possible JSON objects in the generated text
        start_idx = 0
        while True:
            open_brace = partial_json.find("{", start_idx)
            if open_brace < 0:
                break

            # Try all closing braces after this opening brace
            close_brace = partial_json.find("}", open_brace)
            while close_brace >= 0:
                try:
                    candidate = partial_json[open_brace: close_brace + 1]
                    result = json.loads(candidate)

                    # Check if it's valid for our use case
                    if (
                        isinstance(result, dict)
                        and "name" in result
                        and "parameters" in result
                    ):
                        func_names = {fn["name"] for fn in functions}
                        if result["name"] in func_names:
                            print(
                                f"[GEN] ✓ Found valid JSON at step {step}: {result['name']} \nfound by[TID]{threading.get_ident()}"
                            )
                            return result
                except (json.JSONDecodeError, ValueError):
                    pass

                close_brace = partial_json.find("}", close_brace + 1)

            start_idx = open_brace + 1

        # Stop if too much text without finding valid JSON
        if len(partial_json) > 800:
            print(f"[GEN] ✗ Exceeded 800 chars at step {step}")
            raise ValueError("Generated too much text without valid JSON")

    raise ValueError(f"Failed to generate valid JSON after {max_tokens} tokens")
