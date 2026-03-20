"""
evaluation/log_results.py — Appends structured benchmark results to decisions_log.md.

Usage:
    python evaluation/log_results.py --results results/benchmark_tables/run_001.json

This keeps decisions_log.md as the single source of truth.
Never edit results in decisions_log.md manually.
"""

import json
import argparse
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


DECISIONS_LOG = Path(__file__).parent.parent / "decisions_log.md"
RESULTS_MARKER_START = "<!-- RESULTS_START -->"
RESULTS_MARKER_END = "<!-- RESULTS_END -->"


def format_result_block(run: dict) -> str:
    """Format a run result dict as a markdown block for decisions_log.md"""
    r = run.get("retrieval_metrics", {})
    s = run.get("system_metrics", {})
    q = run.get("quality_metrics", {})
    c = run.get("cost", {})
    meta = run.get("metadata", {})

    block = f"""
### RUN-{run['run_id']:03d}
- **Date**: {run.get('date', datetime.utcnow().strftime('%Y-%m-%d %H:%M'))} UTC
- **Architecture**: {run['architecture']}
- **Dataset**: {run['dataset']}
- **Config**: `config.py @ git:{run.get('git_hash', 'unknown')[:8]}`
- **Machine**: {meta.get('machine', 'unknown')} | Python {meta.get('python_version', 'unknown')}

| Metric | Value |
|--------|-------|
| **Recall@5** | {r.get('recall_at_5', 0):.4f} |
| **Precision@5** | {r.get('precision_at_5', 0):.4f} |
| **MRR** | {r.get('mrr', 0):.4f} |
| P50 Latency (s) | {s.get('p50_latency', 0):.3f} |
| P95 Latency (s) | {s.get('p95_latency', 0):.3f} |
| Throughput (q/s) | {s.get('throughput', 0):.1f} |
| Peak RAM (MB) | {s.get('peak_ram_mb', 0):.0f} |
| Storage (MB) | {s.get('storage_mb', 0):.0f} |
| **Faithfulness** | {q.get('faithfulness', 0):.4f} |
| **Answer Relevancy** | {q.get('answer_relevancy', 0):.4f} |
| Context Precision | {q.get('context_precision', 0):.4f} |
| Context Recall | {q.get('context_recall', 0):.4f} |
| Embedding cost ($) | {c.get('embedding', 0):.6f} |
| Generation cost ($) | {c.get('generation', 0):.6f} |
| Eval/judge cost ($) | {c.get('eval', 0):.6f} |
| **Total cost ($)** | {c.get('total', 0):.6f} |
| **Per query ($)** | {c.get('per_query', 0):.6f} |

**Notes**: {run.get('notes', 'No notes.')}

---
"""
    return block


def append_result(result_path: Path) -> None:
    """Load a result JSON and append it to decisions_log.md"""
    with open(result_path, encoding="utf-8") as f:
        run = json.load(f)

    log_text = DECISIONS_LOG.read_text(encoding="utf-8")

    start_idx = log_text.find(RESULTS_MARKER_START)
    end_idx = log_text.find(RESULTS_MARKER_END)

    if start_idx == -1 or end_idx == -1:
        print(f"[log_results] ERROR: Could not find result markers in {DECISIONS_LOG}")
        return

    new_block = format_result_block(run)

    # Insert before the RESULTS_END marker
    new_log = (
        log_text[:end_idx]
        + new_block
        + log_text[end_idx:]
    )

    DECISIONS_LOG.write_text(new_log, encoding="utf-8")
    run_id = run.get('run_id', '?')
    print(f"[log_results] [OK] Appended RUN-{run_id:03d} to {DECISIONS_LOG}")


def save_result(
    run_id: int,
    architecture: str,
    dataset: str,
    retrieval_metrics: dict,
    system_metrics: dict,
    quality_metrics: dict,
    cost: dict,
    notes: str = "",
    git_hash: str = "unknown",
) -> Path:
    """
    Save a benchmark run result to results/benchmark_tables/ and append to decisions_log.md.
    
    This is the main function called by run_benchmark.py after each experiment.
    """
    import platform
    import sys as _sys

    results_dir = Path(__file__).parent.parent / "results" / "benchmark_tables"
    results_dir.mkdir(parents=True, exist_ok=True)

    run = {
        "run_id": run_id,
        "date": datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
        "architecture": architecture,
        "dataset": dataset,
        "retrieval_metrics": retrieval_metrics,
        "system_metrics": system_metrics,
        "quality_metrics": quality_metrics,
        "cost": cost,
        "notes": notes,
        "git_hash": git_hash,
        "metadata": {
            "machine": platform.node(),
            "python_version": _sys.version.split()[0],
            "os": platform.system(),
        }
    }

    out_path = results_dir / f"run_{run_id:03d}_{architecture}_{dataset}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2)

    # Auto-append to decisions_log.md
    append_result(out_path)

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Append benchmark results to decisions_log.md")
    parser.add_argument("--results", required=True, help="Path to result JSON file")
    args = parser.parse_args()
    append_result(Path(args.results))


if __name__ == "__main__":
    main()
