"""
status.py - writes candidates/STATUS.md after every run: is the pipeline alive, what has it found,
and WHY might you not be getting email? Open that file in your repo any time.
"""
import os, time, datetime
import pandas as pd


def is_strong(r):   # keep in sync with notify.py
    wave = (r.get("variability_type") == "likely_pulsating_or_rotating_variable"
            and r.get("ls_r2", 0) > 0.6 and r.get("ls_consistency", 0) > 0.9)
    dip = (r.get("dip_snr", 0) > 20 and r.get("bls_consistency", 0) > 0.9 and r.get("bls_r2", 0) > 0.2)
    return bool(wave or dip)


def main(d="candidates", min_hours=168, fallback_days=14):
    p = lambda n: os.path.join(d, n)

    def read(n):
        try:
            return pd.read_csv(p(n))
        except Exception:
            return pd.DataFrame()

    def lines(n):
        return open(p(n)).read().split() if os.path.exists(p(n)) else []

    now = time.time()
    ranked, leads, dropped = read("ranked_candidates.csv"), read("novel_leads.csv"), read("rechecked_known.csv")
    scanned, seen = len(lines("scanned_tics.txt")), set(lines("notified.txt")) | {t for t in []}
    seen = {x.strip() for x in open(p("notified.txt")).read().split("\n")} if os.path.exists(p("notified.txt")) else set()
    try:
        last = float(open(p("last_notified.txt")).read().strip())
    except Exception:
        last = None
    eq = lambda col: int(ranked[col].astype(str).eq("True").sum()) if col in ranked.columns else 0
    n_flagged, n_novel, n_clean = eq("flagged"), eq("is_novel"), len(leads)
    unseen = leads[~leads["tic"].isin(seen)] if n_clean else leads
    n_pending = int(unseen.apply(lambda r: is_strong(r.to_dict()), axis=1).sum()) if len(unseen) else 0
    last_scan = str(ranked["scanned_at"].max()) if "scanned_at" in ranked.columns and len(ranked) else "never"
    age_days = None
    if last_scan != "never":
        try:
            age_days = (datetime.datetime.utcnow() - datetime.datetime.strptime(last_scan, "%Y-%m-%d %H:%M")).days
        except Exception:
            pass

    due = "now" if last is None else datetime.datetime.utcfromtimestamp(last + min_hours * 3600).strftime("%Y-%m-%d")
    if len(ranked) == 0:
        why = "No stars have been scored yet. Open the Actions tab and check the latest run is green."
    elif age_days is not None and age_days >= 2:
        why = f"Scanning may have STOPPED: the last results were saved {age_days} days ago. Open the Actions tab and look for a red or missing run."
    elif last is not None and now - last < min_hours * 3600:
        why = f"An email went out recently. The next one is allowed after {due}."
    elif n_pending > 0:
        why = "A strong lead is waiting. The next email goes out at the end of the current run."
    elif n_clean > 0:
        why = (f"{n_clean} clean lead(s) exist but none is strong enough for a normal email. "
               f"If nothing strong appears, you get the best of them after {fallback_days} days of silence.")
    else:
        why = "No lead has passed every filter yet. That is normal: the filters are strict on purpose."

    L = ["# Pipeline status\n", f"_Updated {datetime.datetime.utcnow():%Y-%m-%d %H:%M} UTC_\n",
         f"**Why you may not have an email:** {why}\n",
         f"- Stars scanned so far: **{scanned}** (scored: {len(ranked)})",
         f"- Last scan results saved: {last_scan}",
         f"- Flagged by the AI / wave search: {n_flagged}",
         f"- Not in any catalog checked: {n_novel}",
         f"- Also look clean (your leads): **{n_clean}**",
         f"- Strong and not yet emailed: **{n_pending}**",
         f"- Dropped by the live VSX re-check: {len(dropped)}",
         f"- Last email: {'never' if last is None else datetime.datetime.utcfromtimestamp(last):%Y-%m-%d}\n"
         if last is None else
         f"- Last email: {datetime.datetime.utcfromtimestamp(last):%Y-%m-%d}\n"]
    if n_clean:
        top = leads.copy()
        top["priority"] = top[[c for c in ("mean_score", "ls_power") if c in top.columns]].max(axis=1)
        L.append("## Top clean leads\n")
        for _, r in top.sort_values("priority", ascending=False).head(5).iterrows():
            L.append(f"- {r['tic']} - {r.get('likely_type', '')} / {r.get('variability_type', '')}")
    os.makedirs(d, exist_ok=True)
    open(p("STATUS.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
