# logger.py  — new file
import csv, json, time, os
from pathlib import Path

class ExperimentLogger:
    """Writes per-epoch metrics to CSV + a summary JSON."""

    def __init__(self, output_dir: str, experiment_name: str):
        os.makedirs(output_dir, exist_ok=True)
        self.csv_path = Path(output_dir) / f"{experiment_name}_log.csv"
        self.summary_path = Path(output_dir) / f"{experiment_name}_summary.json"
        self._rows = []
        self._start = time.time()
        self._headers_written = False

    def log(self, epoch: int, metrics: dict):
        row = {"epoch": epoch, "elapsed_s": round(time.time() - self._start, 2), **metrics}
        self._rows.append(row)

        # write / append CSV
        write_header = not self._headers_written
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
                self._headers_written = True
            writer.writerow(row)

        # pretty-print to console
        parts = [f"Epoch {epoch:>4}"]
        for k, v in metrics.items():
            parts.append(f"{k}: {v:.5f}" if isinstance(v, float) else f"{k}: {v}")
        print("  ".join(parts))

    def save_summary(self, extra: dict = None):
        summary = {
            "total_epochs": len(self._rows),
            "total_time_s": round(time.time() - self._start, 1),
            "final": self._rows[-1] if self._rows else {},
            "best_val_loss_epoch": min(self._rows, key=lambda r: r.get("val_loss", float("inf")))
                                   .get("epoch") if self._rows else None,
        }
        if extra:
            summary.update(extra)
        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2)