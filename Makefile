VENV = venv
PYTHON = python3.12
FLASK_APP = app.py
REQUIREMENTS = requirements.txt

install:
	$(PYTHON) -m venv $(VENV)
	./$(VENV)/bin/pip install --upgrade pip
	./$(VENV)/bin/pip install -r $(REQUIREMENTS)

run: install
	FLASK_APP=$(FLASK_APP) FLASK_ENV=development ./$(VENV)/bin/flask run --port=3000

clean:
	rm -rf $(VENV)

reinstall: clean install