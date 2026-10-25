"""
notify.py - ONE digest email per week at most (change --min_hours in the workflow to adjust).
Only STRONG leads are included (not in any catalog + very clean repeating shape), max 5, each only once.
The email contains: a picture, copy-paste values for the AAVSO form, and the links you need.
"""
import os, io, json, time, argparse, urllib.parse, urllib.request
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from inspect_candidate import plot_candidate

VSX = "https://vsx.aavso.org"
VSX_SEARCH = "https://www.aavso.org/vsx/index.php?view=search.top"
MAG = 1.0857   # fractional flux change -> magnitudes
SKY_FOV_DEG, SKY_PX = 0.05, 400          # 3 arcminutes wide
HIPS = ["https://alasky.cds.unistra.fr/hips-image-services/hips2fits",
        "https://alaskybis.cds.unistra.fr/hips-image-services/hips2fits"]


def fetch_sky_jpg(ra, dec):
    """Real sky photo (Digitized Sky Survey 2, colour) from the free CDS hips2fits service."""
    q = urllib.parse.urlencode({"hips": "CDS/P/DSS2/color", "width": SKY_PX, "height": SKY_PX,
                                "fov": SKY_FOV_DEG, "ra": ra, "dec": dec, "projection": "TAN",
                                "coordsys": "icrs", "format": "jpg"})
    for ep in HIPS:
        try:
            with urllib.request.urlopen(f"{ep}?{q}", timeout=40) as resp:
                data = resp.read()
            if data[:2] == b"\xff\xd8":
                return data
        except Exception as e:
            print("sky photo fetch failed:", ep, e)
    return None


def make_sky_image(tic, ra, dec, outdir):
    """Saves a sky photo with circles = 1 and 2 TESS pixels (21 and 42 arcsec). Returns path or None."""
    data = fetch_sky_jpg(ra, dec)
    if data is None:
        return None
    try:
        img = mpimg.imread(io.BytesIO(data), format="jpg")
        fig, ax = plt.subplots(figsize=(5, 5))
        half = SKY_FOV_DEG * 3600 / 2          # arcsec
        ax.imshow(img, extent=(half, -half, -half, half), cmap="gray" if img.ndim == 2 else None)  # east left
        ax.add_patch(plt.Circle((0, 0), 21, fill=False, color="yellow", lw=1.6))
        ax.add_patch(plt.Circle((0, 0), 42, fill=False, color="yellow", lw=1.0, ls="--"))
        ax.set_title(f"{tic} - sky photo, 3 arcmin wide (north up, east left)", fontsize=10)
        ax.axis("off")
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, tic.replace(" ", "_") + ".png")
        fig.savefig(path, dpi=80, bbox_inches="tight")
        plt.close(fig)
        return path
    except Exception as e:
        print("sky photo draw failed:", tic, e)
        return None


def is_strong(r):
    wave = (r.get("variability_type") == "likely_pulsating_or_rotating_variable"
            and r.get("ls_r2", 0) > 0.6 and r.get("ls_consistency", 0) > 0.9)
    dip = (r.get("dip_snr", 0) > 20 and r.get("bls_consistency", 0) > 0.9 and r.get("bls_r2", 0) > 0.2)
    return bool(wave or dip)


VSX_API = "https://www.aavso.org/vsx/index.php"


def live_vsx_known(ra, dec, radius_arcsec=42.0):
    """LIVE AAVSO VSX lookup (the VizieR copy used during scanning can be months old).
    Radius 42" = 2 TESS pixels: a known variable that close could be the real source of the signal.
    Returns (True/False/None, name). None = could not check (network/format problem)."""
    q = urllib.parse.urlencode({"view": "api.list", "ra": f"{ra:.6f}", "dec": f"{dec:.6f}",
                                "radius": f"{radius_arcsec / 3600:.6f}", "tomag": "25", "format": "json"})
    try:
        with urllib.request.urlopen(f"{VSX_API}?{q}", timeout=40) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
    except Exception as e:
        print("live VSX check failed:", e)
        return None, None
    names = []

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if str(k).lower() == "name" and isinstance(v, str):
                    names.append(v)
                else:
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(data)
    return (len(names) > 0), (names[0] if names else None)


def hms(x_deg):
    """RA in degrees -> 'HH MM SS.ss' (integer maths, no 60.00 rounding glitches)."""
    t = round(x_deg / 15.0 * 360000) % (24 * 360000)
    h, t = divmod(t, 360000)
    m, t = divmod(t, 6000)
    return f"{h:02d} {m:02d} {t / 100:05.2f}"


def dms(x_deg):
    """Dec in degrees -> '+DD MM SS.s'."""
    sign = "-" if x_deg < 0 else "+"
    t = round(abs(x_deg) * 36000)
    d, t = divmod(t, 36000)
    m, t = divmod(t, 600)
    return f"{sign}{d:02d} {m:02d} {t / 10:04.1f}"


def describe(r):
    """(plain description, VSX type code, period in days, amplitude in mag)"""
    if r["likely_type"] == "likely_eclipsing_binary":
        p, depth = float(r["period_hint"]), float(r["bls_depth"])
        sec = float(r.get("secondary_depth", 0) or 0)
        vt = "EW" if p < 1.0 else ("EA" if sec < 0.5 * depth else "EB")   # best guess; check the picture
        return "eclipsing binary (two stars passing in front of each other)", vt, p, MAG * depth
    if r["variability_type"] == "likely_pulsating_or_rotating_variable":
        return "pulsating or spinning variable star", "VAR", r["ls_period"], MAG * r["ls_amplitude"]
    return "repeating dip (could be a planet or a binary)", "VAR", r["bls_period"], MAG * r["bls_depth"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="candidates")
    ap.add_argument("--max", type=int, default=5)
    ap.add_argument("--skip_live_check", action="store_true", help="Do not re-check live VSX before emailing.")
    ap.add_argument("--min_hours", type=float, default=168, help="Minimum hours between emails (168 = weekly).")
    a = ap.parse_args()
    leads_p = os.path.join(a.dir, "novel_leads.csv")
    note_p = os.path.join(a.dir, "notified.txt")
    last_p = os.path.join(a.dir, "last_notified.txt")
    out = os.environ.get("GITHUB_OUTPUT")

    def done(has_new):
        if out:
            open(out, "a").write(f"has_new={'true' if has_new else 'false'}\n")
        print("has_new =", has_new)

    if os.path.exists(last_p):
        try:
            last = float(open(last_p).read().strip())
        except ValueError:
            last = 0.0
        if time.time() - last < a.min_hours * 3600:
            print("Digest not due yet - leads are saved and will be included in the next one.")
            return done(False)
    if not os.path.exists(leads_p):
        return done(False)
    df = pd.read_csv(leads_p)
    if df.empty:
        return done(False)
    seen = set(open(note_p).read().split("\n")) if os.path.exists(note_p) else set()
    df = df[~df["tic"].isin(seen)]
    df = df[df.apply(lambda r: is_strong(r.to_dict()), axis=1)]
    if df.empty:
        return done(False)
    df["priority"] = df[["mean_score", "ls_power"]].max(axis=1)
    df = df.sort_values("priority", ascending=False)

    # FINAL CHECK right before emailing: drop anything the live VSX already knows.
    keep, dropped, live = [], [], {}
    for _, r in df.iterrows():
        if len(keep) >= a.max:
            break
        if a.skip_live_check:
            known, vname = None, None
        else:
            known, vname = live_vsx_known(float(r["ra"]), float(r["dec"]))
            time.sleep(1)
        if known:
            dropped.append({"tic": r["tic"], "vsx_name": vname,
                            "checked_at": time.strftime("%Y-%m-%d %H:%M", time.gmtime())})
            continue
        live[r["tic"]] = known
        keep.append(r["tic"])
    if dropped:
        dp = os.path.join(a.dir, "rechecked_known.csv")
        old_d = pd.read_csv(dp) if os.path.exists(dp) else pd.DataFrame()
        pd.concat([old_d, pd.DataFrame(dropped)]).to_csv(dp, index=False)
        gone = [d["tic"] for d in dropped]
        open(note_p, "a").write("\n".join(gone) + "\n")                    # never email these
        try:   # bookkeeping must never be able to block the email itself
            lp = pd.read_csv(leads_p)
            lp[~lp["tic"].isin(gone)].to_csv(leads_p, index=False)
            rp = os.path.join(a.dir, "ranked_candidates.csv")                # make LEADS.md honest too
            if os.path.exists(rp):
                rk = pd.read_csv(rp)
                for col in ("already_known", "catalog_type"):
                    rk[col] = rk[col].astype(object) if col in rk.columns else None
                m = rk["tic"].isin(gone)
                rk.loc[m, "already_known"] = True
                rk.loc[m, "catalog_type"] = "VSX-live"
                rk.to_csv(rp, index=False)
        except Exception as e:
            print("bookkeeping after live check failed (email continues):", e)
        print(f"Dropped {len(dropped)} lead(s) already in the live VSX:", ", ".join(gone))
    df = df[df["tic"].isin(keep)]
    if df.empty:
        return done(False)

    repo = os.environ.get("GITHUB_REPOSITORY", "YOU/star-discovery")
    owner = os.environ.get("GITHUB_REPOSITORY_OWNER", "")
    plots = os.path.join(a.dir, "plots")
    L = [f"@{owner} here are your best new leads this week ({len(df)}). Each one is NOT in SIMBAD, AAVSO VSX, "
         "Gaia DR3 variables or ASAS-SN, and shows a clean, repeating shape.\n",
         "## What to do (about 2 minutes per lead)",
         "1. **Look at the picture.** If the bottom two graphs show a thin, clean curve, continue. "
         "If it is a messy cloud, skip it.",
         f"2. **Open [AAVSO VSX]({VSX})** and log in (one-time free registration on that same site).",
         "3. Click **Submit** and choose **New Star Form**. Paste the coordinates from the box below and press "
         "**Spot check**. If it says the star is already known or pending, stop - someone was first. "
         "If not, copy the other values from the box, attach the picture, and submit.",
         "4. AAVSO volunteers review it and may ask you questions. That review is how a find becomes official.\n",
         "## Choosing the star type in the dropdown (use the picture; the moderators can correct it)",
         "- **EA** = flat line between the dips, narrow sharp dips (stars far apart).",
         "- **EB** = brightness never flat, dips of clearly *unequal* depth.",
         "- **EW** = brightness never flat, two *nearly equal* dips per cycle, usually a period under 1 day.",
         "- **ROT** = smooth, slowly changing wave from star spots on a spinning star.",
         "- **VAR** = generic variable. If you are unsure, this is the safe choice.\n"]
    for _, r in df.iterrows():
        png = os.path.join(plots, r["tic"].replace(" ", "_") + ".png")
        if not os.path.exists(png):
            try:
                plot_candidate(r["tic"], plots)
            except Exception as e:
                print("plot failed", r["tic"], e)
        img = f"https://raw.githubusercontent.com/{repo}/main/{png}"
        sky_file = make_sky_image(r["tic"], float(r["ra"]), float(r["dec"]), os.path.join(a.dir, "sky"))
        sky_url = f"https://raw.githubusercontent.com/{repo}/main/{sky_file}" if sky_file else None
        desc, vtype, period, amp = describe(r)
        num = r["tic"].replace("TIC", "").strip()
        simbad = (f"https://simbad.cds.unistra.fr/simbad/sim-coord?Coord={r['ra']}+{r['dec']}"
                  f"&CooFrame=FK5&CooEpoch=2000&Radius=2&Radius.unit=arcmin&submit=submit+query")
        L += [f"---\n### {r['tic']} - best guess: {desc}",
              (f"![{r['tic']}]({img})\n[Open the picture full size]({img})" if os.path.exists(png)
               else "_(Picture could not be drawn for this star - open its page on the TESS link below.)_"),
              (f"![sky photo of {r['tic']}]({sky_url})\n_Real sky photo (DSS2, 3 arcminutes wide, north up). The circles are 1 and 2 TESS pixels. "
               "TESS adds up ALL the light inside them, so if other bright stars sit inside, the signal could come from one of those instead._"
               if sky_url else "_(Sky photo unavailable this time.)_"),
              ("Live VSX check just now: nothing within 42 arcseconds (about 2 TESS pixels)." if live.get(r["tic"]) is False
               else "_Live VSX check could not run just now - rely on the Spot check in the form._"),
              "**Fill the form from top to bottom. Each grey box is ONE field: tap it, copy, paste.**\n",
              "**Primary name**", "```", f"{r['tic']}", "```",
              "**Discoverer** (your own name, the way you want to be credited)\n",
              "**Position - RA** (hours minutes seconds)", "```", hms(float(r['ra'])), "```",
              "**Position - Dec** (degrees arcmin arcsec)", "```", dms(float(r['dec'])), "```",
              f"_If the form wants plain degrees instead: RA {float(r['ra']):.5f}, Dec {float(r['dec']):+.5f}_\n",
              "**Variability type** (pick it in the dropdown)", "```", vtype, "```",
              "**Period (days)**", "```", f"{float(period):.5f}", "```",
              "**Passband** (if TESS is not in the list, leave blank)", "```", "TESS", "```",
              "**Epoch, Max magnitude, Min magnitude:** leave blank if you have no value for them (the pipeline does not measure them).\n",
              "**Remarks**", "```",
              f"Found in NASA TESS 2-minute photometry (TIC {num}) with an automated period search. "
              f"Amplitude about {float(amp):.3f} mag in the TESS band (red-sensitive, so it differs from V-band). "
              f"The period was found automatically and may be half the true period if this is a two-star system.", "```",
              "**Reference / data source**", "```",
              "NASA TESS 2-minute light curves (SPOC, MAST). Position from the TESS Input Catalog (TIC, Gaia-based), J2000.", "```",
              "**Light curve / phase plot upload:** attach the picture above (open it full size, then save it).\n",
              "**Tip:** tick **Draft only** the first time. You can save, look it over, and only then submit for real.\n",
              f"[Open AAVSO VSX]({VSX}) | [VSX coordinate search]({VSX_SEARCH}) | "
              f"[Double-check in SIMBAD]({simbad}) | [TESS page](https://exofop.ipac.caltech.edu/tess/target.php?id={num})"]
        if r["likely_type"] == "likely_planet_transit" and r["variability_type"] != "likely_pulsating_or_rotating_variable":
            L.append("_This one looks like a planet-style dip. The TESS page link is the better place to report it._")
        L.append("")
    open(os.path.join(a.dir, "issue_body.md"), "w").write("\n".join(L))
    open(note_p, "a").write("\n".join(df["tic"]) + "\n")
    open(last_p, "w").write(str(time.time()))
    open(os.path.join(a.dir, "issue_title.txt"), "w").write("Weekly star leads: " + ", ".join(df["tic"]))
    done(True)


if __name__ == "__main__":
    main()
