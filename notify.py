"""
notify.py - so you don't have to check anything: when the pipeline finds a STRONG new lead,
write an issue text (with picture) that the workflow posts to your repo. GitHub then emails you.
"Strong" = not in any catalog + passed the clean-shape test with high marks:
  a very clean repeating wave, or a very clean, high-SNR repeating dip.
Max 3 new leads per run, each lead announced only once (candidates/notified.txt).
"""
import os, argparse
import pandas as pd
from inspect_candidate import plot_candidate


def is_strong(r):
    wave = (r.get("variability_type") == "likely_pulsating_or_rotating_variable"
            and r.get("ls_r2", 0) > 0.6 and r.get("ls_consistency", 0) > 0.9)
    dip = (r.get("dip_snr", 0) > 20 and r.get("bls_consistency", 0) > 0.9 and r.get("bls_r2", 0) > 0.2)
    return bool(wave or dip)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="candidates")
    ap.add_argument("--max", type=int, default=3)
    a = ap.parse_args()
    leads_p = os.path.join(a.dir, "novel_leads.csv")
    note_p = os.path.join(a.dir, "notified.txt")
    out = os.environ.get("GITHUB_OUTPUT")

    def done(has_new):
        if out:
            open(out, "a").write(f"has_new={'true' if has_new else 'false'}\n")
        print("has_new =", has_new)

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
    lines = [f"@{owner} the pipeline found {len(df)} strong new lead(s). Each is NOT in SIMBAD, AAVSO VSX, "
             "Gaia DR3 variables or ASAS-SN, and shows a clean, repeating shape.\n"]
    for _, r in df.iterrows():
        png = os.path.join(plots, r["tic"].replace(" ", "_") + ".png")
        if not os.path.exists(png):
            try:
                plot_candidate(r["tic"], plots)
            except Exception as e:
                print("plot failed", r["tic"], e)
        num = r["tic"].replace("TIC", "").strip()
        simbad = (f"https://simbad.cds.unistra.fr/simbad/sim-coord?Coord={r['ra']}+{r['dec']}"
                  f"&CooFrame=FK5&CooEpoch=2000&Radius=2&Radius.unit=arcmin&submit=submit+query")
        lines += [f"### {r['tic']} - {r['likely_type']} / {r['variability_type']}",
                  f"![{r['tic']}](https://github.com/{repo}/blob/main/{png}?raw=true)",
                  f"- wave period {r['ls_period']:.4f} d | dip period hint {r['period_hint']:.4f} d",
                  f"- [SIMBAD]({simbad}) | [TESS page](https://exofop.ipac.caltech.edu/tess/target.php?id={num})\n"]
    lines += ["**Your 30-second job:** look at the bottom two graphs. A thin clean curve = good. "
              "Then open the SIMBAD link: if the object type is just 'Star' (not Variable/Binary), submit it at "
              "https://www.aavso.org/vsx (Submit new variable). If the picture is a messy cloud, close this issue."]
    open(os.path.join(a.dir, "issue_body.md"), "w").write("\n".join(lines))
    open(note_p, "a").write("\n".join(df["tic"]) + "\n")
    open(os.path.join(a.dir, "issue_title.txt"), "w").write("New star lead: " + ", ".join(df["tic"]))
    done(True)


if __name__ == "__main__":
    main()
