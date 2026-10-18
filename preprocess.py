"""
preprocess.py - turns light curves into model inputs (AstroNet-style):
  * 10-min binning (fast), gentle detrending
  * Box Least Squares period search, then phase-folded GLOBAL and LOCAL views
  * physics-based vetting features (odd/even depth, secondary eclipse, SNR...)
  * Lomb-Scargle search for smooth variability (pulsators, rotators)
  * synthetic-transit augmentation for training

    python preprocess.py --datadir data --outdir processed --force
"""
import os, argparse, warnings
import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares, LombScargle
from tqdm import tqdm

warnings.filterwarnings("ignore")
GLOBAL_BINS, LOCAL_BINS, LOCAL_WINDOW_DURATIONS = 201, 61, 4.0
BIN_MINUTES = 10
BLS_DETREND_POINTS = 145   # ~1 day at 10-min bins
LS_DETREND_POINTS = 433    # ~3 days


def prepare_lightcurve(time, flux, bin_minutes=BIN_MINUTES):
    time, flux = np.asarray(time, float), np.asarray(flux, float)
    ok = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[ok], flux[ok]
    flux = flux / np.median(flux)
    idx = np.floor((time - time[0]) / (bin_minutes / 1440)).astype(np.int64)
    g = pd.DataFrame({"i": idx, "t": time, "f": flux}).groupby("i").mean()
    return g["t"].values, g["f"].values


def clip_outliers(flux, n_sigma=5):
    """Clip UPWARD spikes only (flares, cosmic rays) - never clip eclipses."""
    med = np.median(flux)
    sigma = 1.4826 * np.median(np.abs(flux - med)) + 1e-8
    return np.minimum(flux, med + n_sigma * sigma)


def detrend(flux, window):
    trend = pd.Series(flux).rolling(window=window, center=True, min_periods=1).median().values
    return flux / trend


def bls_search(time, flux, known_period=None, tol=0.05, min_period=0.5, max_period=20.0):
    durations = np.linspace(0.05, 0.3, 10)
    bls = BoxLeastSquares(time, flux)
    periods = None
    if known_period is None:
        try:
            periods = bls.autoperiod(durations, minimum_period=min_period,
                                     maximum_period=max_period, frequency_factor=3.0)
            if len(periods) > 30000:
                periods = periods[np.linspace(0, len(periods) - 1, 30000).astype(int)]
        except Exception:
            periods = None
        if periods is None:
            periods = np.linspace(min_period, max_period, 8000)
    else:
        periods = np.linspace(known_period * (1 - tol), known_period * (1 + tol), 500)
    durations = durations[durations < 0.9 * periods.min()]
    if len(durations) == 0:
        durations = np.array([0.4 * periods.min()])
    r = bls.power(periods, durations)
    i = int(np.argmax(r.power))
    return {"period": float(r.period[i]), "t0": float(r.transit_time[i]),
            "duration": float(r.duration[i]), "depth": float(r.depth[i]),
            "power": float(r.power[i]), "snr": float(r.depth_snr[i])}


def lomb_scargle_search(time, flux, min_period=0.05, max_period=10.0):
    """Finds smooth periodic variability that BLS cannot see."""
    ls = LombScargle(time, flux)
    freq, power = ls.autopower(minimum_frequency=1 / max_period,
                               maximum_frequency=1 / min_period, samples_per_peak=5)
    i = int(np.argmax(power))
    period, p = float(1 / freq[i]), float(power[i])
    try:
        fap = float(ls.false_alarm_probability(p, method="baluev"))
    except Exception:
        fap = float(ls.false_alarm_probability(p))
    model = ls.model(time[0] + np.linspace(0, period, 100), freq[i])
    return {"period": period, "power": p, "fap": fap,
            "amplitude": float(model.max() - model.min())}


def _binned_curve(phase, flux, nbins):
    idx = np.minimum((phase * nbins).astype(int), nbins - 1)
    sums = np.bincount(idx, weights=flux, minlength=nbins)
    cnt = np.bincount(idx, minlength=nbins)
    with np.errstate(invalid="ignore", divide="ignore"):
        return sums / cnt, cnt


def fold_quality(time, flux, period, nbins=40):
    """Automatic 'does the picture look clean?' test.
    r2   = share of the brightness changes explained by the folded curve (messy cloud ~0, clean shape high)
    cons = how well the first half and second half of the data agree when folded (real signals repeat)."""
    ph = ((time - time[0]) % period) / period
    curve, cnt = _binned_curve(ph, flux, nbins)
    filled = np.where(cnt > 0, curve, np.nanmean(curve))
    model = filled[np.minimum((ph * nbins).astype(int), nbins - 1)]
    var = np.var(flux)
    r2 = float(max(0.0, 1 - np.var(flux - model) / var)) if var > 0 else 0.0
    half = time < np.median(time)
    c1, n1 = _binned_curve(ph[half], flux[half], nbins)
    c2, n2 = _binned_curve(ph[~half], flux[~half], nbins)
    both = (n1 > 2) & (n2 > 2)
    cons = float(np.corrcoef(c1[both], c2[both])[0, 1]) if both.sum() >= 8 else 0.0
    return r2, (cons if np.isfinite(cons) else 0.0)


def bin_phase_curve(phase, flux, n_bins, lo, hi):
    edges = np.linspace(lo, hi, n_bins + 1)
    out = np.full(n_bins, np.nan)
    for i in range(n_bins):
        m = (phase >= edges[i]) & (phase < edges[i + 1])
        if m.any():
            out[i] = np.median(flux[m])
    bad = np.isnan(out)
    if bad.all():
        return np.ones(n_bins)
    if bad.any():
        idx = np.arange(n_bins)
        out[bad] = np.interp(idx[bad], idx[~bad], out[~bad])
    return out


def odd_even_depth_ratio(time, flux, period, t0, duration):
    cycle = np.floor((time - t0) / period + 0.5).astype(int)
    ph = ((time - t0 + 0.5 * period) % period) - 0.5 * period
    inn = np.abs(ph) < duration / 2
    odd, even = inn & (cycle % 2 != 0), inn & (cycle % 2 == 0)
    if odd.sum() < 3 or even.sum() < 3:
        return 0.0
    do, de = 1 - np.median(flux[odd]), 1 - np.median(flux[even])
    return float(abs(do - de) / max(abs(do), abs(de), 1e-6))


def secondary_eclipse_depth(phase, flux, duration, period):
    """phase: transit at 0.5, so the secondary eclipse sits at phase 0 / 1."""
    hw = (duration / period) / 2
    d_sec = np.minimum(phase, 1 - phase)
    d_pri = np.abs(phase - 0.5)
    m = d_sec < hw
    base = (d_sec > 3 * hw) & (d_pri > 3 * hw)
    if m.sum() < 3 or base.sum() < 10:
        return 0.0
    return float(max(0.0, np.median(flux[base]) - np.median(flux[m])))


def inject_synthetic_transit(time, flux, rng, period_range=(0.8, 12.0),
                             depth_range=(0.001, 0.02), duration_range=(0.05, 0.25)):
    period = rng.uniform(*period_range)
    depth = rng.uniform(*depth_range)
    dur = rng.uniform(*duration_range)
    t0 = rng.uniform(time.min(), time.min() + period)
    ph = ((time - t0 + 0.5 * period) % period) - 0.5 * period
    out = np.asarray(flux, float).copy()
    out[np.abs(ph) < dur / 2] *= 1 - depth
    return out, period


def process_star(time, flux, known_period=None):
    """Returns (global_view, local_view, vet_features, info)."""
    t, f = prepare_lightcurve(time, flux)
    if len(t) < 300:
        raise ValueError("too few points")
    f = detrend(clip_outliers(f), BLS_DETREND_POINTS)
    b = bls_search(t, f, known_period)
    P, t0, dur = b["period"], b["t0"], b["duration"]

    phase = ((t - t0 + 0.5 * P) % P) / P          # transit at 0.5
    o = np.argsort(phase)
    ps, fs = phase[o], f[o]
    gv = bin_phase_curve(ps, fs, GLOBAL_BINS, 0.0, 1.0)
    hw = (LOCAL_WINDOW_DURATIONS * dur / P) / 2
    lo, hi = 0.5 - hw, 0.5 + hw
    m = (ps > lo) & (ps < hi)
    lv = bin_phase_curve(ps[m], fs[m], LOCAL_BINS, lo, hi) if m.sum() > 5 else np.ones(LOCAL_BINS)

    z = lambda v: (v - np.median(v)) / (np.std(v) + 1e-8)
    vet = np.array([b["power"], b["snr"], b["depth"], dur / P,
                    odd_even_depth_ratio(t, f, P, t0, dur),
                    secondary_eclipse_depth(phase, f, dur, P)], dtype=np.float32)
    return z(gv).astype(np.float32), z(lv).astype(np.float32), vet, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--outdir", default="processed")
    ap.add_argument("--augment_per_negative", type=int, default=2)
    ap.add_argument("--force", action="store_true", help="Reprocess even if output exists.")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    need = ["global_views", "local_views", "vet_features", "y"]
    if not a.force and all(os.path.exists(os.path.join(a.outdir, n + ".npy")) for n in need):
        print(f"{a.outdir}/ already processed - use --force to redo.")
        return
    manifest = pd.read_csv(os.path.join(a.datadir, "manifest.csv"))
    rng = np.random.default_rng(0)
    G, L, V, y, src = [], [], [], [], []

    def add(res, label, source):
        G.append(res[0]); L.append(res[1]); V.append(res[2]); y.append(label); src.append(source)

    for _, r in tqdm(manifest.iterrows(), total=len(manifest), desc="Processing"):
        try:
            d = pd.read_csv(r["file"])
            time, flux = d["time"].values, d["flux"].values
            kp = float(r["known_period"]) if pd.notnull(r.get("known_period")) else None
            add(process_star(time, flux, kp), int(r["label"]), r["file"])
            if r["label"] == 0:
                for _ in range(a.augment_per_negative):
                    inj, p = inject_synthetic_transit(time, flux, rng)
                    add(process_star(time, inj, p), 1, r["file"] + "::synthetic")
        except Exception as e:
            print(f"  skip {r['file']}: {e}")

    V = np.array(V, np.float32)
    mean, std = V.mean(0), V.std(0) + 1e-8
    np.save(os.path.join(a.outdir, "global_views.npy"), np.array(G, np.float32))
    np.save(os.path.join(a.outdir, "local_views.npy"), np.array(L, np.float32))
    np.save(os.path.join(a.outdir, "vet_features.npy"), (V - mean) / std)
    np.save(os.path.join(a.outdir, "vet_mean.npy"), mean)
    np.save(os.path.join(a.outdir, "vet_std.npy"), std)
    np.save(os.path.join(a.outdir, "y.npy"), np.array(y, np.int64))
    pd.DataFrame({"source": src, "label": y}).to_csv(os.path.join(a.outdir, "star_manifest.csv"), index=False)
    y = np.array(y)
    print(f"Saved {len(y)} examples (positive {int((y == 1).sum())}, negative {int((y == 0).sum())})")


if __name__ == "__main__":
    main()
