
install:
	pip install -e .[dev]
	pre-commit install

full:
	pip install -e .[dev,full]

dev:
	pip install -e .[dev,docs]

test:
	pytest -s

cov:
	pytest --cov=meshwell

mypy:
	mypy . --ignore-missing-imports

pylint:
	pylint meshwell

ruff:
	ruff --fix meshwell/*.py

git-rm-merged:
	git branch -D `git branch --merged | grep -v \* | xargs`

update:
	pur

update-pre:
	pre-commit autoupdate --bleeding-edge

git-rm-merged:
	git branch -D `git branch --merged | grep -v \* | xargs`

release:
	git push
	git push origin --tags

build:
	rm -rf dist
	pip install build
	python -m build

jupytext:
	jupytext docs/**/*.ipynb --to py

notebooks:
	jupytext docs/**/*.py --to ipynb

docs:
	jb build docs

.PHONY: drc doc docs

gmsh-ubuntu:
	sudo apt-get install libglu1-mesa
