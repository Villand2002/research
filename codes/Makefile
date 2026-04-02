.PHONY: batch-compare

SIZES ?= 50 100 1000
COUNT ?= 100
ALGORITHMS ?=

batch-compare:
	if [ -x .venv/bin/python ]; then \
		if [ -n "$(ALGORITHMS)" ]; then \
			.venv/bin/python results/run_all_batches_by_size.py --sizes $(SIZES) --count $(COUNT) --algorithms $(ALGORITHMS); \
		else \
			.venv/bin/python results/run_all_batches_by_size.py --sizes $(SIZES) --count $(COUNT); \
		fi; \
	else \
		if [ -n "$(ALGORITHMS)" ]; then \
			python3 results/run_all_batches_by_size.py --sizes $(SIZES) --count $(COUNT) --algorithms $(ALGORITHMS); \
		else \
			python3 results/run_all_batches_by_size.py --sizes $(SIZES) --count $(COUNT); \
		fi; \
	fi
