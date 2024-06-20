default:
	just --list

# create a virtual environment at "./.venv"
make-venv:
	python -m venv .venv

# install `uv`, the blazing fast dependency manager
install-uv:
	python -m pip install uv

# install dependencies
install-deps:
	uv pip install -r requirements.txt

# run the app module
run:
	python -m app
