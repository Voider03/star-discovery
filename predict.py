"""
predict.py - scans TESS stars with TWO independent detectors and checks
whether each hit is already catalogued:
  1. BLS + CNN ensemble  -> transits / eclipsing binaries
  2. Lomb-Scargle        -> smooth pulsating / rotating variables
  3. SIMBAD + AAVSO VSX  -> is it ALREADY known as a variable/binary/planet?

Outputs (in --outdir):
  ranked_candidates.csv  every star scored (accumulates with --append)
  novel_leads.csv        flagged, not catalogued, AND clean-looking -> your leads
  novel_but_messy.csv    flagged + not catalogued but messy fold (probably noise)
  scanned_tics.txt       every star already attempted (never rescanned)

    python predict.py --model_dir model --auto_targets 200 --append
"""
import os, glob, argparse, warnings, datetime
import time as timer
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier

from fetch_data import download_lightcurve, get_real_observed_tic_ids
from preprocess import (process_star, prepare_lightcurve, clip_outliers, detrend,
                        lomb_scargle_search, fold_quality, LS_DETREND_POINTS, BLS_DETREND_POINTS)

warnings.filterwarnings("ignore")
# SIMBAD object-type codes meaning "known variable / binary / planet host".
# (Plain 'Star' entries do NOT count - nearly every TESS target is in SIMBAD.)
KNOWN_CODES = ["V*", "EB*", "SB*", "**", "Ro*", "Pu*", "BY*", "RR*", "Ce*", "Mi*", "Er*",
               "Or*", "Al*", "bL*", "WV*", "El*", "Ir*", "LP*", "dS*", "gD*", "RS*", "Pl"]


def load_ensemble(model_dir):
    paths = sorted(glob.glob(os.path.join(model_dir, "model_*.keras")))
    if not paths:
        raise FileNotFoundError(f"No model_*.keras in {model_dir} - train first.")
    models = [tf.keras.models.load_model(p, compile=False) for p in paths]
    mean, std = np.load(os.path.join(model_dir, "vet_mean.npy")), np.load(os.path.join(model_dir, "vet_std.npy"))
    thr = 0.5
    cfg = os.path.join(model_dir, "ensemble_config.txt")
    if os.path.exists(cfg):
        for line in open(cfg):
            if line.startswith("decision_threshold"):
                thr = float(line.split("=")[1])
    return models, mean, std, thr


def _col(tbl, *names):
    return next((n for n in names if n in tbl.colnames), None)


def check_known(ra, dec, radius_arcsec=10):
    """already_known: True/False/None(unknown). Only variable/binary/planet types count."""
    out = {"already_known": None, "catalog_name": None, "catalog_type": None}
    if ra is None or dec is None:
        return out
    try:
        c = SkyCoord(ra=float(ra) * u.deg, dec=float(dec) * u.deg)
    except Exception:
        return out
    known, answered = False, False
    try:
        s = Simbad()
        s.add_votable_fields("otype")
        r = s.query_region(c, radius=radius_arcsec * u.arcsec)
        answered = True
        if r is not None and len(r) > 0:
            nc, tc = _col(r, "main_id", "MAIN_ID"), _col(r, "otype", "OTYPE")
            for row in r:
                ot = str(row[tc]).strip() if tc else ""
                if any(code in ot for code in KNOWN_CODES):
                    out.update(already_known=True, catalog_name=str(row[nc]), catalog_type=f"SIMBAD:{ot}")
                    return out
    except Exception:
        pass
    try:
        v = Vizier(catalog="B/vsx/vsx", row_limit=5).query_region(c, radius=radius_arcsec * u.arcsec)
        answered = True
        if v and len(v[0]) > 0:
            out.update(already_known=True, catalog_name=str(v[0][0]["Name"]), catalog_type="VSX")
            return out
    except Exception:
        pass
    # Extra catalogs: Gaia DR3 variables, ASAS-SN variables (two lists). Any hit within 5" = already known.
    for cat, label in [("I/358", "Gaia-DR3-variable"), ("II/366", "ASAS-SN"),
                       ("J/MNRAS/519/5271", "ASAS-SN-g")]:
        try:
            v = Vizier(catalog=cat, row_limit=3).query_region(c, radius=5 * u.arcsec)
            answered = True
            if v and any(len(t) > 0 for t in v):
                out.update(already_known=True, catalog_name=label, catalog_type=label)
                return out
        except Exception:
            pass
    out["already_known"] = False if answered else None
    return out


def classify_transit(v):
    depth, oe, sec = v[2], v[4], v[5]
    if sec > 0.005 or oe > 0.3:
        return "likely_eclipsing_binary"
    if depth > 0.01 and oe < 0.15 and sec < 0.002:
        return "likely_planet_transit"
    return "unclassified_periodic_variable"


def classify_variability(ls):
    if ls["fap"] < 1e-4 and ls["amplitude"] > 0.003 and ls["power"] > 0.05:
        return "likely_pulsating_or_rotating_variable"
    return "no_significant_variability"


def score_star(tic, models, vmean, vstd, check_catalogs=True):
    d = download_lightcurve(tic)
    if d is None or len(d[0]) < 1000:
        return None
    time, flux, ra, dec = d
    g, l, v, b = process_star(time, flux)
    x = [g[None, :, None], l[None, :, None], ((v - vmean) / vstd)[None, :]]
    probs = np.array([float(m(x, training=False).numpy().ravel()[0]) for m in models])
    t, f = prepare_lightcurve(time, flux)
    fl = detrend(clip_outliers(f), LS_DETREND_POINTS)
    fb = detrend(clip_outliers(f), BLS_DETREND_POINTS)
    ls = lomb_scargle_search(t, fl)
    ls_r2, ls_cons = fold_quality(t, fl, ls["period"])
    bls_r2, bls_cons = fold_quality(t, fb, b["period"] * (2 if v[4] > 0.3 else 1))
    res = {"mean_score": probs.mean(), "min_score": probs.min(), "max_score": probs.max(),
           "n_agree": int((probs > 0.5).sum()), "n_models": len(probs),
           "bls_period": b["period"],
           "period_hint": b["period"] * (2 if v[4] > 0.3 else 1),  # EBs: BLS often finds HALF the true period
           "bls_snr": float(v[1]), "bls_depth": float(v[2]),
           "odd_even_ratio": float(v[4]), "secondary_depth": float(v[5]),
           "likely_type": classify_transit(v),
           "ls_period": ls["period"], "ls_power": ls["power"], "ls_fap": ls["fap"],
           "ls_amplitude": ls["amplitude"], "variability_type": classify_variability(ls),
           "ls_r2": ls_r2, "ls_consistency": ls_cons, "bls_r2": bls_r2, "bls_consistency": bls_cons,
           "ra": ra, "dec": dec}
    clean_wave = ls["fap"] < 1e-4 and ls_r2 > 0.3 and ls_cons > 0.8
    # proper dip SNR (BLS's own number ignores noise level): depth / (noise / sqrt(points in dip))
    sigma = 1.4826 * np.median(np.abs(np.diff(fb))) / np.sqrt(2) + 1e-9
    ph_t = ((t - b["t0"] + 0.5 * b["period"]) % b["period"]) - 0.5 * b["period"]
    n_in = max(int((np.abs(ph_t) < b["duration"] / 2).sum()), 1)
    dip_snr = float(b["depth"] / (sigma / np.sqrt(n_in)))
    res["dip_snr"] = dip_snr
    clean_dip = dip_snr > 10 and bls_r2 > 0.1 and bls_cons > 0.7
    res["looks_clean"] = bool(clean_wave or clean_dip)
    res.update(check_known(ra, dec) if check_catalogs else
               {"already_known": None, "catalog_name": None, "catalog_type": None})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tic_list", default=None)
    ap.add_argument("--auto_targets", type=int, default=50)
    ap.add_argument("--model_dir", default="model")
    ap.add_argument("--outdir", default="candidates")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--skip_catalog_check", action="store_true")
    ap.add_argument("--require_full_consensus", action="store_true")
    ap.add_argument("--time_budget_min", type=float, default=None,
                    help="Stop scanning after this many minutes and save (for time-limited runners).")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    out_path = os.path.join(a.outdir, "ranked_candidates.csv")
    scanned_path = os.path.join(a.outdir, "scanned_tics.txt")
    models, vmean, vstd, thr = load_ensemble(a.model_dir)
    print(f"Loaded {len(models)} models, decision threshold {thr:.3f}")

    scanned = set(open(scanned_path).read().split("\n")) if os.path.exists(scanned_path) else set()
    if a.tic_list:
        targets = [t.strip() for t in open(a.tic_list) if t.strip() and t.strip() not in scanned]
    else:
        print(f"Picking {a.auto_targets} real, not-yet-scanned TESS targets...")
        targets = get_real_observed_tic_ids(a.auto_targets, exclude_tics=scanned)

    deadline = timer.time() + a.time_budget_min * 60 if a.time_budget_min else float("inf")
    rows = []
    try:
        for tic in tqdm(targets, desc="Scanning"):
            if timer.time() > deadline:
                print("Time budget reached - saving.")
                break
            try:
                r = score_star(tic, models, vmean, vstd, not a.skip_catalog_check)
            except Exception as e:
                print(f"  skip {tic}: {e}")
                r = None
            with open(scanned_path, "a") as fh:
                fh.write(tic + "\n")
            if r:
                r["tic"] = tic
                r["scanned_at"] = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M")
                rows.append(r)
    finally:
        new = pd.DataFrame(rows)
        if not new.empty:
            need = len(models) if a.require_full_consensus else len(models) // 2 + 1
            new["flagged"] = (((new["mean_score"] > thr) & (new["n_agree"] >= need)) |
                              (new["variability_type"] == "likely_pulsating_or_rotating_variable"))
        df = new
        if a.append and os.path.exists(out_path):
            df = pd.concat([pd.read_csv(out_path), new], ignore_index=True).drop_duplicates("tic", keep="last")
        if df.empty:
            print("Nothing scored.")
            return
        df["is_novel"] = df["already_known"].astype(str) == "False"
        df["flagged"] = df["flagged"].astype(str) == "True"
        df = df.sort_values(["flagged", "is_novel", "mean_score"], ascending=[False, False, False])
        df.to_csv(out_path, index=False)
        if "looks_clean" not in df.columns:
            df["looks_clean"] = True
        lc = df["looks_clean"]
        df["looks_clean"] = lc.isna() | (lc.astype(str) == "True")   # older rows without the test are kept
        df.to_csv(out_path, index=False)
        novel = df[df["flagged"] & df["is_novel"]]
        leads = novel[novel["looks_clean"]]
        leads.to_csv(os.path.join(a.outdir, "novel_leads.csv"), index=False)
        novel[~novel["looks_clean"]].to_csv(os.path.join(a.outdir, "novel_but_messy.csv"), index=False)
        print(f"\nScored {len(new)} new stars; {len(df)} total. "
              f"{int(df['flagged'].sum())} flagged, {len(leads)} also look clean and are not catalogued -> novel_leads.csv")
        cols = ["tic", "likely_type", "variability_type", "mean_score", "ls_period", "period_hint", "catalog_type"]
        print(leads[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
