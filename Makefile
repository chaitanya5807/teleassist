.PHONY: setup data index sft-data train eval report serve test lint docker-build

setup:
	python -m pip install -r requirements.txt
	python -m pip install -e .

data:
	@echo "not implemented until Phase 2"

index:
	@echo "not implemented until Phase 3"

sft-data:
	@echo "not implemented until Phase 5"

train:
	@echo "not implemented until Phase 6"

eval:
	@echo "not implemented until Phase 7"

report:
	@echo "not implemented until Phase 7"

serve:
	@echo "not implemented until Phase 8"

test:
	python -m pytest

lint:
	python -m ruff check .

docker-build:
	@echo "not implemented until Phase 8"
