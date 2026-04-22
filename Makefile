NAME=call_me_maybe
COMP=python3
SRC=src/
CC=__pycache__
MYPYCC=.mypy_cache

all: build

build:
	uv run python -m src 

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


