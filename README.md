*This project has been created as part of the 42 curriculum by chrilomb.*

# call me maybe

## Description

**call me maybe** turns natural-language requests into structured function calls
using a small local language model (Qwen/Qwen3-0.6B).

Given a list of available functions (`functions_definition.json`) and a list of
prompts (`function_calling_tests.json`), the program decides for each prompt which
function to call and with which arguments, and writes the result as JSON:

```json
{
  "prompt": "What is the square root of 16?",
  "name": "fn_get_square_root",
  "parameters": {"a": 16.0}
}
```

A 0.6B-parameter model can't be trusted to write valid JSON by itself. The project
therefore uses **constrained decoding**: at every generation step, tokens that
would break the JSON structure or the function schema are masked out, so every
output is guaranteed to parse and to match the schema.

## Instructions

### Requirements

- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/)
- A GPU is used automatically if available (CUDA or MPS); otherwise the program
  runs on CPU.

### Installation

```bash
make install        # uv sync
```

The model weights (~1.5 GB) are downloaded from the Hugging Face Hub on the first
run. On machines with a small home directory, point the caches to a larger disk
first:

```bash
export HF_HOME=/path/with/space/huggingface
export UV_CACHE_DIR=/path/with/space/uv-cache
```

### Running

```bash
make                # uv sync + run on the default input files, timed
```

or directly:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calls.json \
    --workers 1
```

| Option                   | Default                                   | Meaning                          |
|--------------------------|-------------------------------------------|----------------------------------|
| `--functions_definition` | `data/input/functions_definition.json`    | available functions (required)   |
| `--input`                | `data/input/function_calling_tests.json`  | prompts to process               |
| `--output`               | `data/output/function_calls.json`         | where results are written        |
| `--workers`              | `1`                                       | number of threads sharing the model |

### Other targets

```bash
make lint           # flake8 + mypy
make lint-strict    # flake8 + mypy --strict
make clean          # remove __pycache__ / .mypy_cache
```

## Example usage

`data/input/functions_definition.json`:

```json
[
  {
    "name": "fn_greet",
    "description": "Generate a greeting message for a person by name.",
    "parameters": {"name": {"type": "string"}},
    "returns": {"type": "string"}
  },
  {
    "name": "fn_get_square_root",
    "description": "Calculate the square root of a number.",
    "parameters": {"a": {"type": "number"}},
    "returns": {"type": "number"}
  }
]
```

`data/input/function_calling_tests.json`:

```json
[
  {"prompt": "Greet john"},
  {"prompt": "Calculate the square root of 144"}
]
```

Output (`data/output/function_calls.json`):

```json
[
  {"prompt": "Greet john", "name": "fn_greet", "parameters": {"name": "john"}},
  {"prompt": "Calculate the square root of 144", "name": "fn_get_square_root",
   "parameters": {"a": 144.0}}
]
```

If a prompt can't be processed, its entry contains an `"error"` message instead
and the program carries on with the other prompts.

## Algorithm explanation

The code is in `src/generator.py`; `generate_function_call()` is the entry point.

### 1. Token table

The vocabulary from `tokenizer.json` stores tokens in GPT-2's byte-level form
(`Ġ` = space, `Ċ` = newline). `decode_token()` maps each entry back to the text it
represents. This is done once for the whole vocabulary (~151k tokens), and
boolean masks are precomputed for each kind of position in the JSON:

| Mask          | Tokens allowed                                         |
|---------------|--------------------------------------------------------|
| `string_body` | anything without `"` or a newline                      |
| `string_end`  | tokens starting with `"` (close the string)            |
| `number_body` | tokens made only of `0-9`, `.`, `-`                    |
| `number_end`  | tokens starting with `,` or `}` (close the number)     |

Special tokens (`<|im_end|>`, `<think>`, …) are excluded from every mask. The
table is built once and shared by all worker threads behind a lock.

### 2. Prompt

The prompt uses Qwen's chat format and lists each function on one line:

```
- fn_greet(name: string): Generate a greeting message for a person by name.
```

It ends with an empty `<think></think>` block, which turns off Qwen3's
"thinking" mode so the model answers straight away.

### 3. Forcing the structure

Every part of the output that is fixed by the schema is appended to the token
sequence directly, **without calling the model**:

```
{"name": "  <name>  ", "parameters": {"a": <value>, "b": "<value>"}}
└─forced─┘  model  └──────forced──────┘     model  └forced┘  model
```

The model is only asked for the decisions that are really open: which function,
and the value of each argument.

### 4. Choosing the function

At each step, the only allowed tokens are those that keep the text a prefix of
at least one real function name. Names are matched by prefix, so the model can
never invent a function. As soon as only one name is still possible, the rest of
it is filled in without asking the model.

### 5. Generating each argument

Arguments are generated in the order they appear in the definition, using the
mask that matches their type:

- **string** — the opening `"` is forced, then tokens from `string_body` are
  picked until the model chooses a `string_end` token. An escaped `\"` is kept
  as part of the string. The result is JSON-unescaped.
- **number / integer** — tokens from `number_body`, and each candidate is also
  checked against a regular expression (`-?\d*\.?\d*`), so `1.2.3` or `--5` can't
  appear. `number_end` is only allowed once the text is a complete number.
  Integers can't contain `.`. Numbers are returned as `float`, integers as `int`.
- **boolean** — the logits of `true` and `false` are compared.

At each step, the logits of disallowed tokens are effectively set to −∞, and the
highest-scoring remaining token is picked (greedy decoding).

## Design decisions

- **Force everything the schema already fixes.** The SDK only exposes
  `get_logits_from_input_ids()`, with no cache between calls, so each step
  re-processes the whole sequence. Not asking the model for the JSON skeleton is
  both the guarantee of validity and the biggest speed gain.
- **Short prompt.** For the same reason, every prompt token is paid for at every
  step. The functions are listed in one line each instead of as indented JSON.
- **Precomputed numpy masks.** Masking 151k logits with a boolean array and
  taking `argmax` costs almost nothing compared to a forward pass.
- **Greedy decoding.** The task is extraction, not creative writing; the most
  likely valid token is what we want, and the output is reproducible.
- **Threads sharing one model.** `ThreadPoolExecutor` lets several prompts run
  at once without loading the model several times. The token table is built
  once behind a lock.
- **One error per prompt, not per run.** Any exception during a prompt is caught
  and written as an `"error"` entry, so one bad prompt never crashes the program.
- **No pydantic classes.** The generator uses plain functions instead of
  classes, since the subject requires every class to use pydantic.

## Performance analysis

- **Validity:** 100% of outputs are valid JSON that matches the schema, by
  construction. The decoder can only produce tokens that keep the output valid,
  and the closing structure is always added by the code.
- **Accuracy:** <!-- TODO: fill in, e.g. "9/9 correct on the provided tests" -->
- **Speed:** <!-- TODO: fill in the time printed by `make` -->
  The number of model calls per prompt is roughly *(tokens needed to choose the
  name) + (tokens in the argument values) + 1 per argument*. Free generation also
  had to produce every brace, quote and key.
- **Limits:** strings are capped at 64 tokens and numbers at 24 tokens. Strings
  can't contain newlines. Multi-byte characters that a tokenizer splits across
  several tokens are not produced.

## Challenges faced

- **Speed.** The first version let the model write freely and searched the
  output for a JSON object. It took about 10 minutes for the test set, over the
  5-minute limit. Each step re-runs the model over the whole sequence, so the cost
  is the number of forward passes multiplied by the sequence length. Forcing the
  structure and shortening the prompt addressed both.
- **Byte-level vocabulary.** Vocab entries such as `Ġhello` aren't the text they
  stand for. Rebuilding GPT-2's byte-to-unicode table was needed to know what
  each token actually writes.
- **Ending strings and numbers.** A token can contain both content and a
  terminator (`",`, `}`). Treating any token *starting* with the terminator as
  "end of value", and adding the closing structure ourselves, avoids mixing the
  two. Escaped quotes (`\"`) needed special handling so they don't end the string.
- **Logit size vs. vocab size.** The model returns more logits than
  `tokenizer.json` has entries, so the masks are padded to the logit length.

## Testing strategy

- Running the full pipeline on `data/input/function_calling_tests.json` and
  checking each output by hand against the expected function and arguments.
- A fake model (returning logits that favour a fixed target string) to test the
  decoder logic without loading Qwen: numbers, floats, regex strings, escaped
  quotes.
- Checking that every output file parses with `json.loads` and that each
  `parameters` object has exactly the keys and types from the definition.
- Error cases: missing input file, invalid JSON input, empty function list.
- `make lint` / `make lint-strict` (flake8 + mypy) on every change.

## Resources

- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Hugging Face — Tokenizers: byte-level BPE](https://huggingface.co/learn/nlp-course/chapter6/5)
- [OpenAI GPT-2 `encoder.py` (`bytes_to_unicode`)](https://github.com/openai/gpt-2/blob/master/src/encoder.py)
- [Willard & Louf, *Efficient Guided Generation for Large Language Models* (2023)](https://arxiv.org/abs/2307.09702)
- [JSON specification (RFC 8259)](https://www.rfc-editor.org/rfc/rfc8259)
- [Python `concurrent.futures`](https://docs.python.org/3/library/concurrent.futures.html)
- [NumPy documentation](https://numpy.org/doc/)

### Use of AI

Claude (Anthropic) was used during the project for:

- **Performance:** analysing why the run took ~10 minutes, and rewriting
  `src/generator.py` into the forced-structure constrained decoder described
  above (token masks, name prefix matching, typed value generation).
- **Typing:** fixing `mypy --strict` errors (numpy array type parameters) and
  silencing errors coming from the provided `llm_sdk`.
- **Documentation:** drafting this README.

All AI-generated code was reviewed, tested and adjusted by hand.
