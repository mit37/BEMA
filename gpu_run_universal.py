"""
One command for a rented GPU (built for an RTX 5090): trains and scores BEMA-U, the
unseen-category model, on the same frozen inputs Jev answered.

  python gpu_run_universal.py            # everything (about 5-6 hours on a 5090)
  python gpu_run_universal.py --quick    # 10-minute smoke test of every stage
  python gpu_run_universal.py --stages bi_base,cross_base   # only some stages
  python gpu_run_universal.py --max-hours 6                  # skip stages that would start after 6 h

What it does:
  1. checks the GPU and that PyTorch supports it (an RTX 5090 needs a CUDA 12.8+ build)
  2. rebuilds the frozen test inputs and checks them against inputs_manifest.json
  3. unpacks the training data shipped in data/universal/*.xz and checks data/universal/manifest.json
  4. trains and scores each stage below, logging to gpu_run_universal.log
  5. zips every prediction file into universal_results.zip -- send that file back

Nothing here calls Jev or any paid API. Stop the rented machine when it prints DONE.
"""
import hashlib
import json
import lzma
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile

LOG = "gpu_run_universal.log"
QUICK = "--quick" in sys.argv
ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}

# name -> settings. "inj" adds the Claude-written planted-note training messages.
STAGES = {
    "bi_base":         {"U_MODE": "bi", "U_MODEL_ID": "BAAI/bge-base-en-v1.5", "U_EPOCHS": "2", "U_LR": "3e-5"},
    "cross_base":      {"U_MODE": "cross", "U_MODEL_ID": "BAAI/bge-reranker-base", "U_EPOCHS": "1", "U_LR": "2e-5"},
    "cross_base_inj":  {"U_MODE": "cross", "U_MODEL_ID": "BAAI/bge-reranker-base", "U_EPOCHS": "1", "U_LR": "2e-5", "U_INJECT": "1"},
    "bi_large_inj":    {"U_MODE": "bi", "U_MODEL_ID": "BAAI/bge-large-en-v1.5", "U_EPOCHS": "2", "U_LR": "2e-5", "U_INJECT": "1"},
    "cross_large_inj": {"U_MODE": "cross", "U_MODEL_ID": "BAAI/bge-reranker-large", "U_EPOCHS": "1", "U_LR": "1e-5", "U_INJECT": "1"},
    "cross_base_inj_s2": {"U_MODE": "cross", "U_MODEL_ID": "BAAI/bge-reranker-base", "U_EPOCHS": "1", "U_LR": "2e-5", "U_INJECT": "1", "U_SEED": "2"},
    "cross_base_inj_s3": {"U_MODE": "cross", "U_MODEL_ID": "BAAI/bge-reranker-base", "U_EPOCHS": "1", "U_LR": "2e-5", "U_INJECT": "1", "U_SEED": "3"},
}


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def run(args, extra=None, label=None):
    t0 = time.time()
    log(f"\n=== {label or ' '.join(args)}")
    with open(LOG, "a", encoding="utf-8") as f:
        p = subprocess.run([sys.executable, *args], env={**ENV, **(extra or {})}, stdout=f, stderr=subprocess.STDOUT)
    log(f"    done in {time.time() - t0:.0f}s (exit {p.returncode})")
    if p.returncode != 0:
        log(f"FAILED: {label}. Send {LOG} in the chat.")
        sys.exit(1)


def check_gpu():
    import torch
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "cuda": torch.cuda.is_available()}
    if not info["cuda"]:
        log(json.dumps(info))
        log("No CUDA GPU visible to PyTorch. Stop here and send gpu_run_universal.log in the chat.")
        sys.exit(1)
    cap = torch.cuda.get_device_capability(0)
    info.update({"gpu": torch.cuda.get_device_name(0), "capability": f"{cap[0]}.{cap[1]}",
                 "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)})
    log(json.dumps(info))
    if cap[0] >= 12 and tuple(int(x) for x in (torch.version.cuda or "0.0").split(".")[:2]) < (12, 8):
        log("This GPU (Blackwell, e.g. RTX 5090) needs a PyTorch build for CUDA 12.8 or newer. Run:\n"
            "  pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128\n"
            "then start this script again.")
        sys.exit(1)
    try:   # a real kernel launch fails here if the build lacks this GPU's architecture
        (torch.ones(256, 256, device="cuda", dtype=torch.bfloat16) @ torch.ones(256, 256, device="cuda", dtype=torch.bfloat16)).sum().item()
    except Exception as e:
        log(f"PyTorch cannot run on this GPU ({e}). Reinstall torch for CUDA 12.8+ (see above).")
        sys.exit(1)
    return info


def unpack_data():
    for name in ("train.jsonl", "val.jsonl", "injection_train.jsonl"):
        src, dst = f"data/universal/{name}.xz", f"data/universal/{name}"
        if os.path.exists(src) and not os.path.exists(dst):
            with lzma.open(src, "rb") as a, open(dst, "wb") as b:
                shutil.copyfileobj(a, b)
    man = json.load(open("data/universal/manifest.json"))
    bad = [p for p, w in man.items()
           if not os.path.exists(p) or hashlib.sha256(open(p, "rb").read()).hexdigest() != w["sha256"]]
    if bad:
        log(f"Training data does not match the manifest: {bad}. Send {LOG} in the chat.")
        sys.exit(1)
    log(f"Training data matches the manifest ({len(man)} files).")


def main():
    open(LOG, "w").close()
    t_start = time.time()
    info = check_gpu()
    sys.path.insert(0, ".")
    import gpu_run as G   # reuse the frozen-input rebuild and check from the first GPU run
    G.LOG = LOG
    run(["prepare_multitask.py"], label="rebuild the dataset splits")
    run(["bench_jev.py", "prepare"], label="rebuild the frozen test inputs")
    run(["redteam_jev.py", "build"], label="rebuild the red-team inputs")
    G.check_inputs()
    unpack_data()

    want = list(STAGES)
    if "--stages" in sys.argv:
        want = sys.argv[sys.argv.index("--stages") + 1].split(",")
    if QUICK:
        want = ["bi_base", "cross_base_inj"]
    max_h = float(sys.argv[sys.argv.index("--max-hours") + 1]) if "--max-hours" in sys.argv else 9.0
    done = []
    for stage in want:
        elapsed = (time.time() - t_start) / 3600
        if elapsed > max_h:
            log(f"skipping {stage}: {elapsed:.1f} h elapsed, over --max-hours {max_h}")
            continue
        env = {**STAGES[stage], "U_NAME": f"bu_{stage}"}
        if QUICK:
            env.update({"U_MAX_STEPS": "30", "U_EPOCHS": "1", "U_LIMIT": "200"})
        run(["bema_universal.py", "train"], env, f"train {stage}")
        run(["bema_universal.py", "predict"], env, f"score {stage}")
        done.append(stage)
        shutil.rmtree(f"bu_{stage}_model", ignore_errors=True) if os.environ.get("U_KEEP_MODELS") != "1" else None
        log(f"    total elapsed {(time.time() - t_start) / 3600:.2f} h")

    files = []
    for d in ("runs/bema", "runs/redteam/bema", "runs/universal/bema"):
        if os.path.isdir(d):
            files += [os.path.join(d, f) for f in os.listdir(d) if "bu_" in f]
    files += ["runs/universal/models.json", LOG]
    out = "universal_results_quick.zip" if QUICK else "universal_results.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            if os.path.exists(f):
                z.write(f)
        z.writestr("gpu_env.json", json.dumps({**info, "stages": done,
                                               "hours": round((time.time() - t_start) / 3600, 2)}, indent=1))
    log(f"\nDONE in {(time.time() - t_start) / 3600:.2f} h. Download {os.path.abspath(out)} "
        f"({len(files)} files), send it in the chat, then STOP the rented machine.")


if __name__ == "__main__":
    main()
