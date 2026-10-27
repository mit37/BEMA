# Raw benchmark results

`runs_raw.tar.xz` holds the whole `runs/` folder from the Jev benchmark (mock dry runs excluded), including the laptop-GPU predictions and run log:
frozen inputs with per-text SHA-256, every raw Jev and LLM API response, every BEMA prediction,
the generated reports, and the independent audits (`runs/audit/`).

    tar -xJf results_archive/runs_raw.tar.xz      # recreates runs/
    python3 results_summary.py                    # regenerates runs/results_summary.json and runs/RESULTS.md

`runs_raw.tar.xz.sha256` is the archive's checksum. No API keys are stored anywhere in it.
