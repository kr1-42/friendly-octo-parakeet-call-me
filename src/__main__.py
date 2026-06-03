import argparse
import json
import sys
import multiprocessing as mp
from functools import partial
from llm_sdk import Small_LLM_Model as bob
from . import generator

def load_vocab(model: bob) -> dict[int, str]:
    """Returns a dict of {token_id: token_string}."""
    path = model.get_path_to_tokenizer_file()  # this gives tokenizer.json

    with open(path, "r") as f:
        tokenizer = json.load(f)

    # vocab is {"token_string": id, ...} — we invert it to {id: token_string}
    raw_vocab: dict[str, int] = tokenizer["model"]["vocab"]
    id_to_token: dict[int, str] = {v: k for k, v in raw_vocab.items()}

    # also add the special tokens (added_tokens section)
    for entry in tokenizer.get("added_tokens", []):
        id_to_token[entry["id"]] = entry["content"]

    return id_to_token

def process_item(item: dict, functions: list[dict]) -> dict:
    """Process a single item. Pool will handle parallelization."""
    prompt = item["prompt"]

    try:
        model = bob()
        vocab = load_vocab(model)
        result = generator.generate_function_call(model, prompt, functions, vocab)
        print(f"[PID {mp.current_process().pid}] ✓ {prompt[:50]}")
        return {
            "prompt": prompt,
            "name": result["name"],
            "parameters": result["parameters"]
        }
    except Exception as e:
        print(f"[PID {mp.current_process().pid}] ✗ {prompt[:50]} → {e}", file=sys.stderr)
        return {"prompt": prompt, "error": str(e)}


def split_into_chunks(lst: list, n: int) -> list[list]:
    """Split list into n roughly equal chunks."""
    k, remainder = divmod(len(lst), n)
    chunks = []
    start = 0
    for i in range(n):
        end = start + k + (1 if i < remainder else 0)
        chunks.append(lst[start:end])
        start = end
    return chunks


def load_json_file(path: str) -> list:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in {path}: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM function calling")
    parser.add_argument("--functions_definition", default="data/input/functions_definition.json")
    parser.add_argument("--input", default="data/input/function_calling_tests.json")
    parser.add_argument("--output", default="data/output/function_calls.json")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    functions = load_json_file(args.functions_definition)
    prompts = load_json_file(args.input)

    num_workers = min(args.workers, len(prompts))  # don't spawn more than needed

    print(f"Processing {len(prompts)} prompts across {num_workers} workers...")

    # Use Pool with maxTasksPerChild to recycle workers and save memory
    with mp.Pool(processes=num_workers, maxtasksperchild=100) as pool:
        process_fn = partial(process_item, functions=functions)
        all_results = pool.map(process_fn, prompts)

    # preserve original order
    prompt_order = {item["prompt"]: i for i, item in enumerate(prompts)}
    all_results.sort(key=lambda r: prompt_order.get(r["prompt"], 999))

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Done. {len(all_results)}/{len(prompts)} results written to {args.output}")


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)  # fork works better on Linux for pickling
    main()
