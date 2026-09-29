NAME=call_me_maybe
COMP=python3
SRC=src/
CC=__pycache__
MYPYCC=.mypy_cache

all: build

build:
	uv sync
	uv run python -m src \
	--functions_definition data/input/functions_definition.json \
	--input data/input/function_calling_tests.json \
	--output data/output/function_calls.json \
	--workers 2

lint:
	flake8 .
	mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	flake8 .
	mypy . --strict

clean:
	rm -rf $(NAME)
	rm -rf $(CC)
	rm -rf $(SRC)/$(CC)
	rm -rf $(MYPYCC)


