VENV = venv
PYTHON = python3.12
FLASK_APP = app.py
REQUIREMENTS = requirements.txt

install:
	$(PYTHON) -m venv $(VENV)
	./$(VENV)/bin/python -m pip install --upgrade pip --break-system-packages
	./$(VENV)/bin/python -m pip install -r $(REQUIREMENTS)

run: install
	./$(VENV)/bin/python -m flask run --port=3000 --host=0.0.0.0

clean:
	rm -rf $(VENV)

reinstall: clean install
