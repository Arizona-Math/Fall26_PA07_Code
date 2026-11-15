.PHONY: check experiments submission
check:
	python3 tests/check.py
experiments:
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts/experiments.py
submission:
	python3 -c 'import zipfile; z=zipfile.ZipFile("submission.zip","w",zipfile.ZIP_DEFLATED); z.write("spectral.py"); z.close()'
