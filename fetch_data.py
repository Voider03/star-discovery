"""
fetch_data.py - downloads REAL NASA TESS light curves (via MAST) for training.

Positives: host stars of confirmed transiting planets (NASA Exoplanet Archive).
Negatives: random, actually-observed TESS 2-minute targets.
Resumable: progress is saved in data/manifest.csv, rerun any time.

    python fetch_data.py --n_positive 150 --n_negative 300 --outdir data
"""
import os, argparse, time, warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import lightkurve as lk
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from astroquery.mast import Observations

warnings.filterwarnings("ignore")
MAX_SECTORS = 4          # sectors stitched per star (keeps downloads/BLS fast)
MAX_SECTOR_NUMBER = 90   # sectors 1..90 all exist by now


def _clean_tic(name):
    s = str(name).strip()
    return s if s.upper().startswith("TIC") else f"TIC {s}"


def get_real_observed_tic_ids(n, exclude_tics=None, rng=None):
    """TIC IDs that TESS really observed at 2-min cadence (SPOC), sampled
    across random sectors. Never guesses random IDs."""
    exclude = set(exclude_tics or [])
    rng = rng or np.random.default_rng()
    per_sector = max(20, n // 3 + 1)
    picked = []
    for sector in rng.permutation(np.arange(1, MAX_SECTOR_NUMBER + 1)):
        if len(picked) >= n:
            break
        try:
            obs = Observations.query_criteria(
                obs_collection="TESS", dataproduct_type="timeseries",
                provenance_name="SPOC", sequence_number=int(sector))
        except Exception:
            continue
        if obs is None or len(obs) == 0:
            continue
        names = list({_clean_tic(x) for x in obs["target_name"]})
        rng.shuffle(names)
        taken = 0
        for tic in names:
            if tic in exclude or tic in picked:
                continue
            picked.append(tic)
            taken += 1
            if len(picked) >= n or taken >= per_sector:
                break
    return picked


def download_lightcurve(tic, max_sectors=MAX_SECTORS):
    """Returns (time, flux, ra, dec) or None."""
    try:
        search = lk.search_lightcurve(tic, mission="TESS", author="SPOC", exptime=120)
        if len(search) == 0:
            return None
        coll = search[:max_sectors].download_all()
        if coll is None or len(coll) == 0:
            return None
        meta = coll[0].meta
        lc = coll.stitch().remove_nans().normalize()
        return (np.asarray(lc.time.value, float), np.asarray(lc.flux.value, float),
                meta.get("RA"), meta.get("DEC"))
    except Exception:
        return None


def get_confirmed_planet_hosts(limit):
    print("Querying NASA Exoplanet Archive for confirmed transiting planets...")
    table = NasaExoplanetArchive.query_criteria(
        table="pscomppars",
        select="pl_name,tic_id,pl_orbper,pl_tranmid",
        where="tran_flag=1 and tic_id is not null and pl_orbper is not null")
    df = table.to_pandas().drop_duplicates(subset="tic_id").sample(frac=1, random_state=0)
    return df.head(limit)


def load_manifest(outdir):
    p = os.path.join(outdir, "manifest.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        return df[df["file"].apply(os.path.exists)].to_dict("records")
    return []


def save_manifest(rows, outdir):
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "manifest.csv"), index=False)


def _save(tic, d, label, outdir, rows, **extra):
    fname = os.path.join(outdir, f"{'positive' if label else 'negative'}_{tic.replace(' ', '_')}.csv")
    pd.DataFrame({"time": d[0], "flux": d[1]}).to_csv(fname, index=False)
    rows.append({"file": fname, "label": label, "tic": tic,
                 "known_period": extra.get("known_period"), "known_epoch": extra.get("known_epoch")})
    save_manifest(rows, outdir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_positive", type=int, default=150)
    ap.add_argument("--n_negative", type=int, default=300)
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--time_budget_min", type=float, default=None,
                    help="Stop downloading after this many minutes (resume later).")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    deadline = time.time() + a.time_budget_min * 60 if a.time_budget_min else float("inf")
    rows = load_manifest(a.outdir)
    done = {r["tic"] for r in rows}

    have = sum(r["label"] == 1 for r in rows)
    if have < a.n_positive:
        hosts = get_confirmed_planet_hosts(a.n_positive * 4)
        for _, h in tqdm(hosts.iterrows(), total=len(hosts), desc="Positives"):
            if have >= a.n_positive or time.time() > deadline:
                break
            tic = _clean_tic(h["tic_id"])
            if tic in done:
                continue
            done.add(tic)
            d = download_lightcurve(tic)
            if d is None or len(d[0]) < 1000:
                continue
            _save(tic, d, 1, a.outdir, rows, known_period=float(h["pl_orbper"]))
            have += 1

    have = sum(r["label"] == 0 for r in rows)
    if have < a.n_negative and time.time() < deadline:
        for tic in tqdm(get_real_observed_tic_ids((a.n_negative - have) * 3, exclude_tics=done),
                        desc="Negatives"):
            if have >= a.n_negative or time.time() > deadline:
                break
            d = download_lightcurve(tic)
            if d is None or len(d[0]) < 1000:
                continue
            _save(tic, d, 0, a.outdir, rows)
            have += 1

    npos = sum(r["label"] == 1 for r in rows)
    print(f"Done: {npos} positive, {len(rows) - npos} negative light curves in {a.outdir}/")


if __name__ == "__main__":
    main()
