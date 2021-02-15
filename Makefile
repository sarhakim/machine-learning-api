SHELL = /bin/bash
install :
	poetry install --no-root

activate_venv :
	source .venv/bin/activate

run_api :
	python ml_api/api.py --port 5000

create_table :
	python ml_api/database.py

delete_data :
	python -c 'from ml_api.database import delete_data; delete_data()'
