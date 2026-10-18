"""
inspect_candidate.py - plot a candidate so it can be judged by eye.
    python inspect_candidate.py "TIC 12345678"
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fetch_data import download_lightcurve
from preprocess import (prepare_lightcurve, clip_outliers, detrend, bls_search,
                        lomb_scargle_search, BLS_DETREND_POINTS, LS_DETREND_POINTS)


def fold(t, f, P, t0):
    ph = ((t - t0 + 0.5 * P) % P) / P - 0.5
    o = np.argsort(ph)
    return ph[o], f[o]


def plot_candidate(tic, outdir="candidate_plots"):
    """Saves a 3-panel PNG; returns {'path','bls','ls'} or None if no data."""
    d = download_lightcurve(tic)
    if d is None:
        return None
    t, f = prepare_lightcurve(d[0], d[1])
    fb = detrend(clip_outliers(f), BLS_DETREND_POINTS)
    fl = detrend(clip_outliers(f), LS_DETREND_POINTS)
    b, ls = bls_search(t, fb), lomb_scargle_search(t, fl)
    fig, ax = plt.subplots(3, 1, figsize=(9, 9))
    ax[0].plot(t, f, ".", ms=1)
    ax[0].set_title(f"{tic} - raw brightness over time")
    ph, ff = fold(t, fb, b["period"], b["t0"])
    ax[1].plot(ph, ff, ".", ms=2)
    ax[1].set_title(f"Dip search, folded at {b['period']:.3f} d (depth {b['depth']*100:.2f}%, SNR {b['snr']:.0f})")
    ph, ff = fold(t, fl, ls["period"], t[0])
    ax[2].plot(ph, ff, ".", ms=2)
    ax[2].set_title(f"Wave search, folded at {ls['period']:.3f} d (size {ls['amplitude']*100:.2f}%)")
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, tic.replace(" ", "_") + ".png")
    plt.savefig(out, dpi=90)
    plt.close(fig)
    return {"path": out, "bls": b, "ls": ls}


def main():
    r = plot_candidate(sys.argv[1])
    if r is None:
        sys.exit("No 2-min TESS light curve found for " + sys.argv[1])
    print("Saved", r["path"])


if __name__ == "__main__":
    main()
