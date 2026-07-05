.PHONY: dev ingest eval eval-adv online-eval test reindex ui build graph-build finetune-mine finetune-train ab-report drift synth-eval

dev:
	docker-compose up -d && uvicorn api.main:app --reload

ui:
	cd ui && npm run dev

ingest:
	python -m ingestion.pipeline --source ./data

eval:
	python eval/run_eval.py --dataset eval/golden_dataset/questions.jsonl

eval-adv:
	python eval/run_eval.py --dataset eval/golden_dataset/adversarial.jsonl

online-eval:
	python observability/online_eval.py

test:
	pytest tests/ -x --cov

reindex:
	python scripts/rebuild_index.py --blue-green && \
	python -c "import asyncio; from api.caching import semantic_cache; \
	asyncio.run(semantic_cache.invalidate_all())"

build:
	docker build -f infra/docker/Dockerfile.api -t mra-api .
	docker build -f infra/docker/Dockerfile.ingestion -t mra-ingestion .
	docker build -f infra/docker/Dockerfile.ui -t mra-ui .

graph-build:
	python -m ingestion.indexing.graph_builder

finetune-mine:
	python finetuning/mine_pairs.py

finetune-train:
	python finetuning/train_biencoder.py --pairs finetuning/pairs.jsonl

ab-report:
	python eval/ab_report.py

drift:
	python observability/drift_monitor.py

synth-eval:
	python eval/golden_dataset/generate_synthetic.py --n 100
