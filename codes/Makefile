.PHONY: batch-compare

SIZES ?= 50 100 1000
COUNT ?= 100

batch-compare:
	if [ -x .venv/bin/python ]; then .venv/bin/python results/run_all_batches_by_size.py --sizes $(SIZES) --count $(COUNT); else python3 results/run_all_batches_by_size.py --sizes $(SIZES) --count $(COUNT); fi
