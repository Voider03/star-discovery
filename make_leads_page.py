"""
make_leads_page.py - turns candidates/novel_leads.csv into candidates/LEADS.md:
a scrollable page with a picture for each top lead. Open LEADS.md on GitHub and just look.
Runs automatically in the workflow after every scan.
"""
import os, argparse
import pandas as pd
from inspect_candidate import plot_candidate

HEADER = """# Your leads - just look at the pictures

Each lead is a star NOT catalogued in SIMBAD / AAVSO VSX as a variable or binary.
Look at the **bottom two graphs** of each picture:

- **GOOD:** the dots form a clean, thin curve (a dip or a smooth wave). The thinner the curve, the better.
- **BAD:** the dots are a messy cloud with no clear shape. Ignore these.

If a picture looks GOOD: click the SIMBAD link, and if the star shows no variable/binary type,
submit it at https://www.aavso.org/vsx (Submit new variable).

"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leads", default="candidates/novel_leads.csv")
    ap.add_argument("--outdir", default="candidates")
    ap.add_argument("--max_new", type=int, default=10, help="New pictures to draw per run.")
    ap.add_argument("--show", type=int, default=50, help="Leads shown on the page.")
    a = ap.parse_args()
    page = os.path.join(a.outdir, "LEADS.md")
    plots = os.path.join(a.outdir, "plots")
    os.makedirs(plots, exist_ok=True)
    if not os.path.exists(a.leads):
        open(page, "w").write(HEADER + "No leads yet - check back after a scan finishes.\n")
        return
    df = pd.read_csv(a.leads)
    if df.empty:
        open(page, "w").write(HEADER + "No leads yet - check back after the next scan.\n")
        return
    df["priority"] = df[["mean_score", "ls_power"]].max(axis=1)
    df = df.sort_values("priority", ascending=False)

    new, lines = 0, [HEADER, f"{len(df)} leads total; showing the top {a.show}.\n"]
    shown = 0
    for _, r in df.iterrows():
        if shown >= a.show:
            break
        png = os.path.join(plots, r["tic"].replace(" ", "_") + ".png")
        if not os.path.exists(png):
            if new >= a.max_new:
                continue
            new += 1
            try:
                if plot_candidate(r["tic"], plots) is None:
                    continue
            except Exception as e:
                print("plot failed", r["tic"], e)
                continue
        shown += 1
        num = r["tic"].replace("TIC", "").strip()
        simbad = (f"https://simbad.cds.unistra.fr/simbad/sim-coord?Coord={r['ra']}+{r['dec']}"
                  f"&CooFrame=FK5&CooEpoch=2000&Radius=2&Radius.unit=arcmin&submit=submit+query")
        lines.append(f"## {r['tic']}  -  {r['likely_type']} / {r['variability_type']}\n")
        lines.append(f"- AI score: {r['mean_score']:.2f} | wave period: {r['ls_period']:.3f} d | "
                     f"dip period hint: {r['period_hint']:.3f} d\n"
                     f"- [Check in SIMBAD]({simbad}) | "
                     f"[TESS page](https://exofop.ipac.caltech.edu/tess/target.php?id={num})\n")
        lines.append(f"![{r['tic']}](plots/{os.path.basename(png)})\n")
    open(page, "w").write("\n".join(lines))
    print(f"LEADS.md written: {shown} leads shown, {new} new pictures drawn")


if __name__ == "__main__":
    main()
