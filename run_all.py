"""Reproduce every result: data extraction, rolling-origin experiments, evaluation, decision analysis and costs.

Run from this folder. The SEC archives must be in data_raw/sec (see src/download_sec.py).
Runtime on a 64-core CPU: several hours, almost all of it in run_experiments.py.
"""
import subprocess, sys, os
os.environ.setdefault('GBM_THREADS', '10')
steps = [
    ['src/extract_xbrl.py'], ['src/extract_submissions.py'], ['src/build_panel.py'], ['src/features.py'],
    ['src/run_experiments.py', '--groups', 'qcop,qcop_cur,qpoint,qcop_earlier,gbm,surface,surface_cur,gbm_cur,simple', '--workers', '6'],
    ['src/evaluate.py'], ['src/decision.py'], ['src/robustness.py'], ['src/filing_audit.py'], ['src/cost.py'], ['src/export_models.py'],
    ['src/interval_coverage.py'], ['src/download_etf.py'], ['src/validate_etf.py'], ['src/check_reproduction.py'],
]
for s in steps:
    print('>>', ' '.join(s), flush=True)
    subprocess.run([sys.executable] + s, check=True)
