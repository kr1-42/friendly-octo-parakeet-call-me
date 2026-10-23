NAME=call_me_maybe
COMP=python3
SRC=src/
CC=__pycache__
MYPYCC=.mypy_cache

all: build

install:
	uv sync

build:
	uv sync
	time uv run python -m src \
	--functions_definition data/input/functions_definition.json \
	--input data/input/function_calling_tests.json \
	--output data/output/function_calls.json \
	--workers 1

lint:
	flake8 src/
	mypy src/ --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	flake8 src/
	mypy src/ --strict

clean:
	rm -rf $(NAME)
	rm -rf $(CC)
	rm -rf $(SRC)/$(CC)
	rm -rf $(MYPYCC)


