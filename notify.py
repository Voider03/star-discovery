"""
notify.py - ONE digest email per week at most (change --min_hours in the workflow to adjust).
Only STRONG leads are included (not in any catalog + very clean repeating shape), max 5, each only once.
The email contains: a picture, copy-paste values for the AAVSO form, and the links you need.
"""
import os, time, argparse
import pandas as pd
from inspect_candidate import plot_candidate

VSX = "https://vsx.aavso.org"
VSX_SEARCH = "https://www.aavso.org/vsx/index.php?view=search.top"
MAG = 1.0857   # fractional flux change -> magnitudes


def is_strong(r):
    wave = (r.get("variability_type") == "likely_pulsating_or_rotating_variable"
            and r.get("ls_r2", 0) > 0.6 and r.get("ls_consistency", 0) > 0.9)
    dip = (r.get("dip_snr", 0) > 20 and r.get("bls_consistency", 0) > 0.9 and r.get("bls_r2", 0) > 0.2)
    return bool(wave or dip)


def describe(r):
    """(plain description, VSX type code, period in days, amplitude in mag)"""
    if r["likely_type"] == "likely_eclipsing_binary":
        return "eclipsing binary (two stars passing in front of each other)", "E", r["period_hint"], MAG * r["bls_depth"]
    if r["variability_type"] == "likely_pulsating_or_rotating_variable":
        return "pulsating or spinning variable star", "VAR", r["ls_period"], MAG * r["ls_amplitude"]
    return "repeating dip (could be a planet or a binary)", "VAR", r["bls_period"], MAG * r["bls_depth"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="candidates")
    ap.add_argument("--max", type=int, default=5)
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
    df = df.sort_values("priority", ascending=False).head(a.max)

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
         "4. AAVSO volunteers review it and may ask you questions. That review is how a find becomes official.\n"]
    for _, r in df.iterrows():
        png = os.path.join(plots, r["tic"].replace(" ", "_") + ".png")
        if not os.path.exists(png):
            try:
                plot_candidate(r["tic"], plots)
            except Exception as e:
                print("plot failed", r["tic"], e)
        desc, vtype, period, amp = describe(r)
        num = r["tic"].replace("TIC", "").strip()
        simbad = (f"https://simbad.cds.unistra.fr/simbad/sim-coord?Coord={r['ra']}+{r['dec']}"
                  f"&CooFrame=FK5&CooEpoch=2000&Radius=2&Radius.unit=arcmin&submit=submit+query")
        L += [f"---\n### {r['tic']} - best guess: {desc}",
              f"![{r['tic']}](https://github.com/{repo}/blob/main/{png}?raw=true)",
              "Copy-paste values for the form:",
              "```",
              f"Name / ID:        {r['tic']} (TESS)",
              f"RA  (deg, J2000): {float(r['ra']):.5f}",
              f"Dec (deg, J2000): {float(r['dec']):.5f}",
              f"Type (best guess): {vtype}",
              f"Period (days):    {float(period):.5f}",
              f"Amplitude:        about {float(amp):.3f} mag (TESS band)",
              "Source:           NASA TESS 2-minute light curves (MAST), found by an automated search",
              "```",
              f"[Open AAVSO VSX]({VSX}) | [VSX coordinate search]({VSX_SEARCH}) | "
              f"[Double-check in SIMBAD]({simbad}) | [TESS page](https://exofop.ipac.caltech.edu/tess/target.php?id={num})"]
        if r["likely_type"] == "likely_planet_transit":
            L.append("_This one looks like a planet-style dip. The TESS page link is the better place to report it._")
        L.append("")
    open(os.path.join(a.dir, "issue_body.md"), "w").write("\n".join(L))
    open(note_p, "a").write("\n".join(df["tic"]) + "\n")
    open(last_p, "w").write(str(time.time()))
    open(os.path.join(a.dir, "issue_title.txt"), "w").write("Weekly star leads: " + ", ".join(df["tic"]))
    done(True)


if __name__ == "__main__":
    main()
