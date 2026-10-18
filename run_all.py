"""
run_all.py - one command: fetch -> preprocess -> train (-> scan).

    python run_all.py                # fetch + preprocess + train
    python run_all.py --scan         # ...then scan stars too
    python run_all.py --skip_fetch   # reuse data/
"""
import argparse, subprocess, sys


def run(cmd):
    print("\n$", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_positive", type=int, default=150)
    ap.add_argument("--n_negative", type=int, default=300)
    ap.add_argument("--augment_per_negative", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n_models", type=int, default=5)
    ap.add_argument("--min_precision", type=float, default=0.9)
    ap.add_argument("--fetch_budget_min", type=float, default=None)
    ap.add_argument("--skip_fetch", action="store_true")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--tic_list", default=None)
    a = ap.parse_args()
    py = sys.executable
    if not a.skip_fetch:
        cmd = [py, "fetch_data.py", "--n_positive", str(a.n_positive), "--n_negative", str(a.n_negative), "--outdir", "data"]
        if a.fetch_budget_min:
            cmd += ["--time_budget_min", str(a.fetch_budget_min)]
        run(cmd)
    run([py, "preprocess.py", "--datadir", "data", "--outdir", "processed",
         "--augment_per_negative", str(a.augment_per_negative), "--force"])
    run([py, "train_model.py", "--processed", "processed", "--epochs", str(a.epochs),
         "--n_models", str(a.n_models), "--min_precision", str(a.min_precision), "--outdir", "model"])
    if a.scan or a.tic_list:
        cmd = [py, "predict.py", "--model_dir", "model", "--append"]
        if a.tic_list:
            cmd += ["--tic_list", a.tic_list]
        run(cmd)
    print("\nDone. Model in model/. Scan with: python predict.py --model_dir model --append")


if __name__ == "__main__":
    main()
