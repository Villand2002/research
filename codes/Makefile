.PHONY: batch-compare

SIZES ?= 50 100 1000
COUNT ?= 100

batch-compare:
	source venv/bin/activate  && python3 results/run_all_batches_by_size.py --sizes $(SIZES) --count $(COUNT)
