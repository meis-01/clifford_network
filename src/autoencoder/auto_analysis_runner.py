#!/usr/bin/env python3
import os
import time
import subprocess
from datetime import datetime

RESULTS_DIR = "runs/weight_init_comparison"
METHODS = ["trabelsi", "structured_preserve"]
CHECK_FILES = [os.path.join(RESULTS_DIR, f"init_{m}", "summary.json") for m in METHODS]
LOG_PATH = os.path.join(RESULTS_DIR, "auto_analysis.log")
DONE_FLAG = os.path.join(RESULTS_DIR, "analysis_done")

SLEEP_SECONDS = 60


def log(msg):
    ts = datetime.now().isoformat()
    line = f"{ts} - {msg}\n"
    print(line, end="")
    try:
        with open(LOG_PATH, "a") as f:
            f.write(line)
    except Exception:
        pass


if __name__ == '__main__':
    log("auto_analysis_runner started")

    if os.path.exists(DONE_FLAG):
        log("Done flag present — exiting")
        raise SystemExit(0)

    while True:
        all_done = True
        for cf in CHECK_FILES:
            if not os.path.exists(cf):
                all_done = False
                break
        if all_done:
            log("Detected summary files for all methods — running analysis")
            cmd = [
                "python", "-u", "-m", "src.autoencoder.comparison_analysis",
                "--results-dir", RESULTS_DIR,
                "--methods",
            ] + METHODS
            try:
                with open(LOG_PATH, "a") as f:
                    f.write(f"{datetime.now().isoformat()} - running: {' '.join(cmd)}\n")
                    proc = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(__file__)), stdout=f, stderr=f)
                    f.write(f"{datetime.now().isoformat()} - analysis exit code: {proc.returncode}\n")
                if proc.returncode == 0:
                    open(DONE_FLAG, "w").write(datetime.now().isoformat())
                    log("Analysis completed successfully; created analysis_done flag")
                else:
                    log(f"Analysis finished with exit code {proc.returncode}")
            except Exception as e:
                log(f"Error running analysis: {e}")
            break

        log(f"Not ready yet — missing files. Sleeping {SLEEP_SECONDS}s")
        time.sleep(SLEEP_SECONDS)
