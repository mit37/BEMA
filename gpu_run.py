"""
One-shot GPU run for a local machine (Windows, Linux or Mac).

Rebuilds the exact frozen test inputs, checks them against inputs_manifest.json
(so the GPU results are provably on the same text as everything else), trains
the larger BEMA models on the GPU, scores them on every input, and zips the
prediction files into gpu_results.zip to send back.

What it trains (training splits only; test splits are only scored):
  bema_v2_full_gpu   all-MiniLM-L6-v2 (22.7M), every layer, 3 epochs   [the cloud run of this was stopped in epoch 3]
  bema_v3            BAAI/bge-base-en-v1.5 (110M), every layer, 4 epochs
  zs3_from_clinc / zs3_from_banking
                     bge-base zero-shot transfer: trained on one dataset's
                     intents, tested on the other's unseen intents

Usage (from the repo folder, inside the virtual environment):
  python gpu_run.py            # needs an NVIDIA GPU (CUDA) or Apple GPU (MPS)
  python gpu_run.py --quick    # 5-minute smoke test of the whole pipeline
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import zipfile

ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
QUICK = "--quick" in sys.argv
LOG = "gpu_run.log"


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def run(args, extra_env=None, label=None):
    t0 = time.time()
    log(f"\n=== {label or ' '.join(args)}")
    env = {**ENV, **(extra_env or {})}
    with open(LOG, "a", encoding="utf-8") as f:
        p = subprocess.run([sys.executable, *args], env=env, stdout=f, stderr=subprocess.STDOUT)
    log(f"    done in {time.time() - t0:.0f}s (exit {p.returncode})")
    if p.returncode != 0:
        log(f"FAILED: {label}. See {LOG}. Send gpu_run.log in the chat.")
        sys.exit(1)


def check_gpu():
    import torch
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
            "cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available()}
    if info["cuda"]:
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    log(json.dumps(info))
    if not (info["cuda"] or info["mps"]):
        log("No GPU visible to PyTorch. Stop here and send gpu_run.log in the chat.")
        sys.exit(1)
    return info


def check_inputs():
    man = json.load(open("inputs_manifest.json", encoding="utf-8"))
    bad = []
    for path, want in man.items():
        if path.endswith("categories.json"):
            got = hashlib.sha256(open(path, "rb").read()).hexdigest()
            if got != want["sha256"]:
                bad.append(path)
            continue
        hs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                hs.append(r.get("text_sha256") or r.get("state_sha256"))
        if len(hs) != want["lines"] or hashlib.sha256("\n".join(hs).encode()).hexdigest() != want["sha256_of_text_hashes"]:
            bad.append(path)
    if bad:
        log(f"Input mismatch in {len(bad)} files, e.g. {bad[:3]}. Stop and send gpu_run.log.")
        sys.exit(1)
    log(f"All {len(man)} input files match the cloud run exactly.")


def main():
    open(LOG, "w").close()
    info = check_gpu()
    run(["prepare_multitask.py"], label="rebuild the dataset splits")
    run(["bench_jev.py", "prepare"], label="rebuild the frozen test inputs")
    run(["redteam_jev.py", "build"], label="rebuild the red-team inputs")
    check_inputs()

    q = {"V2_MAX_STEPS": "5", "V2_EPOCHS": "1"} if QUICK else {}
    v2 = {"V2_FREEZE": "0", "V2_EPOCHS": "3", "V2_LR": "5e-5", "V2_NAME": "bema_v2_full_gpu",
          "V2_CKPT": "bema_v2_full_gpu.pt", **q}
    run(["bema_v2.py", "train"], v2, "train BEMA v2 full (MiniLM, all layers, 3 epochs)")
    run(["bema_v2.py", "predict"], v2, "score BEMA v2 full")
    run(["bema_spell.py"], v2, "score BEMA v2 full + speller")

    v3 = {"V2_MODEL_ID": "BAAI/bge-base-en-v1.5", "V2_FREEZE": "0", "V2_EPOCHS": "4", "V2_LR": "3e-5",
          "V2_NAME": "bema_v3", "V2_CKPT": "bema_v3.pt", **q}
    run(["bema_v2.py", "train"], v3, "train BEMA v3 (bge-base 110M, all layers, 4 epochs)")
    run(["bema_v2.py", "predict"], v3, "score BEMA v3")
    run(["bema_spell.py"], v3, "score BEMA v3 + speller")

    zs = {"ZS_MODEL_ID": "BAAI/bge-base-en-v1.5", "ZS_POOL": "cls", "ZS_PREFIX": "zs3", "ZS_FREEZE": "0",
          "ZS_EPOCHS": "2", "ZS_SKIP_OFFSHELF": "1", **({"ZS_MAX_STEPS": "5", "ZS_EPOCHS": "1"} if QUICK else {})}
    run(["bema_zeroshot.py", "train"], zs, "train zero-shot transfer (bge-base, 2 epochs each way)")
    run(["bema_zeroshot.py", "predict"], zs, "score zero-shot transfer")

    names = ("bema_v2_full_gpu", "bema_v3", "zs3_from_clinc", "zs3_from_banking")
    files = [os.path.join(d, f) for d in ("runs/bema", "runs/redteam/bema") for f in os.listdir(d)
             if any(n in f for n in names)]
    files += ["runs/bema/timing.json", "runs/bema/zeroshot_temperatures.json", LOG]
    out = "gpu_results_quick.zip" if QUICK else "gpu_results.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            if os.path.exists(f):
                z.write(f)
        z.writestr("gpu_env.json", json.dumps(info, indent=1))
    log(f"\nDONE. Send {os.path.abspath(out)} in the chat ({len(files)} files).")


if __name__ == "__main__":
    main()
