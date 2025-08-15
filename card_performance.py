#!/usr/bin/env python3
r"""
Commander Card Performance (Top 10 by recent conversion) + Theoretical Best Deck (Moxfield Export)
-----------------------------------------------------------------------------------------------

What this script does
---------------------
1) Pull the last N months of EDH tournaments (>= participant_min) from TopDeck.gg.
2) Compute commander conversion rates to Top 16 over that window and list the top 10.
3) **Eligibility dial:** require each commander to have at least **M** entries over the window (default **M = 5 × months**; configurable via `--min-deck-entries`).
4) You pick a commander (CLI flag or interactive prompt).
5) The script pulls **all decklists** for that commander over the same period.
6) It classifies each deck’s performance and computes **per-card performance deltas**:
   - prevalence overall
   - prevalence among **Top16** decks ("best")
   - prevalence among **bottom-half** decks ("worst")
   - **lift = prevalence_top16 - prevalence_bottom_half** (positive = good card signal)
   - optional Fisher exact test p-values if SciPy is installed
7) Outputs:
   - `decks_<commander>.csv` (one row per deck appearance, with metadata)
   - `card_performance_<scope>.csv` (one row per card with metrics; scope = commander or GLOBAL)
8) **Theoretical Best Deck Builder (Moxfield)†**
   - `--best-deck` builds a decklist from the highest‑lift cards for the **selected commander**.
   - `--best-deck-global` builds from **all eligible commanders combined**.
   - `--best-deck-size` (default 99) and `--best-deck-min-prev` (default 0.15) control selection.
   - Writes a **paste‑ready** Moxfield text file: `best_deck_<scope>_moxfield.txt` with `Commander` + `Deck` sections.

† Deck legality nuances (e.g., color identity) are not enforced here; this is a pure "strongest cards" list.

Requirements
------------
- TopDeck.gg API key with standings+decklist access via `--api-key` or env `TOPDECK_API_KEY`.
- Python 3.9+; SciPy optional for p‑values.
"""

from __future__ import annotations
import os
import re
import json
import argparse
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Iterable

import requests
import pandas as pd
import numpy as np

# Optional for significance testing. If unavailable, we skip p-values gracefully.
try:
    from scipy.stats import fisher_exact  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

API_BASE = "https://topdeck.gg/api"
SESSION = requests.Session()

# -----------------------------
# API helpers
# -----------------------------

def td_headers(api_key: str) -> Dict[str, str]:
    return {"Authorization": api_key, "Content-Type": "application/json"}


def list_recent_edh_tournaments(api_key: str, last_days: int, participant_min: int) -> List[dict]:
    """POST /v2/tournaments with filters, return completed EDH events.
    Includes: id, name, startDate (unix seconds), participants.
    """
    url = f"{API_BASE}/v2/tournaments"
    body = {
        "game": "Magic: The Gathering",
        "format": "EDH",
        "last": last_days,
        "participantMin": participant_min,
        # Request columns we rely on downstream
        "columns": [
            "decklist",
            "wins",
            "losses",
            "draws",
            "winRate",
            "id",
            "standing",
        ],
    }
    r = SESSION.post(url, headers=td_headers(api_key), json=body, timeout=60)
    r.raise_for_status()
    data = r.json()
    tours = []
    for t in data:
        tid = t.get("id") or t.get("_id") or t.get("TID")
        if not tid:
            continue
        status = str(t.get("status", "Completed")).lower()
        start = t.get("startDate")
        if start is None:
            continue
        if status.startswith("complet"):
            tours.append(
                {
                    "id": tid,
                    "name": t.get("name"),
                    "startDate": start,  # unix seconds
                    "participants": t.get("participants") or t.get("participantCount"),
                }
            )
    return tours


def fetch_standings(api_key: str, tid: str) -> List[dict]:
    """GET /v2/tournaments/{TID}/standings"""
    url = f"{API_BASE}/v2/tournaments/{tid}/standings"
    r = SESSION.get(url, headers=td_headers(api_key), timeout=60)
    if r.status_code == 404:
        return []
    r.raise_for_status()
    return r.json()

# -----------------------------
# Decklist parsing
# -----------------------------

SECTION_HEADERS = (
    "Commanders",
    "Commander",
    "Mainboard",
    "Sideboard",
    "Companion",
    "Maybeboard",
)

def normalize_cmdr_key(commanders: Iterable[str]) -> str:
    names = [str(c).strip() for c in commanders if str(c).strip()]
    key = " / ".join(sorted(names))
    return key.replace("’", "'").replace("‘", "'")


def extract_commanders_from_deckobj(deckobj: Optional[dict]) -> List[str]:
    if not deckobj or not isinstance(deckobj, dict):
        return []
    commanders = deckobj.get("Commanders") or deckobj.get("Commander") or {}
    if isinstance(commanders, dict):
        return list(commanders.keys())
    return []


def extract_cards_from_deckobj(deckobj: Optional[dict]) -> Dict[str, int]:
    """Return all **non-commander** cards (name -> qty) from deckObj sections."""
    cards: Dict[str, int] = {}
    if not deckobj or not isinstance(deckobj, dict):
        return cards
    for sec in ("Mainboard", "Sideboard", "Companion", "Maybeboard"):
        block = deckobj.get(sec)
        if isinstance(block, dict):
            for name, qty in block.items():
                try:
                    q = int(qty)
                except Exception:
                    q = 1
                if not name:
                    continue
                name_n = str(name).replace("’", "'").replace("‘", "'").strip()
                cards[name_n] = cards.get(name_n, 0) + max(1, q)
    return cards

PLAIN_QTY_RE = re.compile(r"^(\d+)\s+(.+?)\s*$")

def parse_plaintext_decklist(decklist: str) -> Tuple[List[str], Dict[str, int]]:
    commanders: List[str] = []
    cards: Dict[str, int] = {}
    if not decklist:
        return commanders, cards
    parts = re.split(r"~~(.*?)~~", decklist)
    cur_header: Optional[str] = None
    for i, token in enumerate(parts):
        if i % 2 == 1:
            cur_header = token.strip(); continue
        content = token
        if not cur_header or not content:
            continue
        lines = [ln.strip() for ln in content.strip().splitlines() if ln.strip()]
        if cur_header in ("Commanders", "Commander"):
            for ln in lines:
                m = PLAIN_QTY_RE.match(ln)
                name = (m.group(2) if m else ln).replace("’", "'").replace("‘", "'").strip()
                if name:
                    commanders.append(name)
        elif cur_header in ("Mainboard", "Sideboard", "Companion", "Maybeboard"):
            for ln in lines:
                m = PLAIN_QTY_RE.match(ln)
                if m:
                    q = int(m.group(1))
                    name = m.group(2).replace("’", "'").replace("‘", "'").strip()
                    if name:
                        cards[name] = cards.get(name, 0) + max(1, q)
    return commanders, cards

# -----------------------------
# Data building
# -----------------------------

def build_rows(api_key: str, last_days: int, participant_min: int) -> pd.DataFrame:
    tours = list_recent_edh_tournaments(api_key, last_days, participant_min)
    rows: List[dict] = []
    for t in tours:
        stand = fetch_standings(api_key, t["id"]) or []
        for s in stand:
            standing = s.get("standing")
            if standing is None:
                continue
            deckobj = s.get("deckObj")
            decklist = s.get("decklist") or ""
            commanders = extract_commanders_from_deckobj(deckobj)
            cards: Dict[str, int] = {}
            if not commanders:
                commanders, cards = parse_plaintext_decklist(decklist)
            else:
                cards = extract_cards_from_deckobj(deckobj)
            if not commanders:
                continue
            cmd_key = normalize_cmdr_key(commanders)
            rows.append(
                {
                    "tournament_id": t["id"],
                    "tournament_name": t.get("name"),
                    "tournament_date": datetime.fromtimestamp(t["startDate"], tz=timezone.utc).date(),
                    "participants": t.get("participants"),
                    "standing": int(standing),
                    "commander": cmd_key,
                    "cards_json": json.dumps(cards, ensure_ascii=False),
                    "has_deck": bool(cards),
                }
            )
    return pd.DataFrame(rows)

# -----------------------------
# Commander ranking (conversion)
# -----------------------------

def top_commanders_by_conversion(df: pd.DataFrame, weeks_back: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["commander", "entries", "top16", "conversion_rate"])
    df = df.copy()
    df["week"] = pd.to_datetime(df["tournament_date"]) - pd.to_timedelta(
        pd.to_datetime(df["tournament_date"]).dt.weekday, unit="D"
    )
    weeks_sorted = sorted(df["week"].unique())
    if len(weeks_sorted) > weeks_back:
        weeks_sorted = weeks_sorted[-weeks_back:]
    df = df[df["week"].isin(weeks_sorted)].copy()

    g_entries = df.groupby(["week", "commander"], as_index=False).agg(entries=("standing", "count"))
    g_top16 = (
        df[df["standing"] <= 16]
        .groupby(["week", "commander"], as_index=False)
        .agg(top16=("standing", "count"))
    )
    m = g_entries.merge(g_top16, on=["week", "commander"], how="left").fillna({"top16": 0})

    by_cmd = (
        m.groupby("commander")[ ["top16", "entries"] ]  # pandas 2.x: list, not tuple
        .sum()
        .reset_index()
    )
    by_cmd["conversion_rate"] = by_cmd.apply(
        lambda r: r["top16"] / r["entries"] if r["entries"] > 0 else np.nan,
        axis=1,
    )
    by_cmd = by_cmd.sort_values("conversion_rate", ascending=False)
    return by_cmd

# -----------------------------
# Card performance math
# -----------------------------

def classify_deck_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Label decks as Top16 (best) and Bottom-half (worst) with robust fallbacks."""
    df = df.copy()
    df["top16"] = (df["standing"] <= 16).astype(int)

    # Tournament-relative percentile when participants are known (not always available)
    part = pd.to_numeric(df.get("participants"), errors="coerce")
    df["participants_num"] = part
    df["standing_pct"] = np.where(
        df["participants_num"].notna() & (df["participants_num"] > 0),
        df["standing"].astype(float) / df["participants_num"].astype(float),
        np.nan,
    )
    df["bottom_half"] = (df["standing_pct"] >= 0.5).astype(float)  # 1.0/0.0/NaN

    # Fallback A: if no bottom_half==1.0, try per-tournament median of available rows
    if not (df["bottom_half"] == 1.0).any():
        df["t_median"] = df.groupby("tournament_id")["standing"].transform("median")
        df["bottom_half"] = (df["standing"].astype(float) >= df["t_median"].astype(float)).astype(int)

    # Fallback B: if still no bottom sample, split within-commander by rank percentile
    if df["bottom_half"].sum() == 0:
        df["rank_pct_cmd"] = df["standing"].rank(pct=True, method="max")
        df["bottom_half"] = (df["rank_pct_cmd"] >= 0.5).astype(int)

    df["bottom_half"] = df["bottom_half"].fillna(0).astype(int)
    return df


def compute_card_performance(df_commander: pd.DataFrame) -> pd.DataFrame:
    df = df_commander.copy()
    df["cards"] = df["cards_json"].apply(lambda s: json.loads(s) if s else {})

    total_decks = len(df)
    n_top = int(df["top16"].sum())
    n_bot = int(df["bottom_half"].sum())

    if n_bot == 0:
        cutoff = df["standing"].quantile(0.5)
        df["bottom_half"] = (df["standing"] >= cutoff).astype(int)
        n_bot = int(df["bottom_half"].sum())

    card_stats: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        cards: Dict[str, int] = row["cards"]
        if not isinstance(cards, dict):
            continue
        present_cards = set(cards.keys())
        for name in present_cards:
            st = card_stats.setdefault(name, {"count_all": 0, "count_top": 0, "count_bot": 0})
            st["count_all"] += 1
            if row["top16"] == 1:
                st["count_top"] += 1
            if row["bottom_half"] == 1:
                st["count_bot"] += 1

    rows = []
    n_all = float(total_decks)
    n_top_f = float(n_top) if n_top else np.nan
    n_bot_f = float(n_bot) if n_bot else np.nan

    for name, st in card_stats.items():
        prev_all = (st["count_all"] / n_all) if n_all else np.nan
        prev_top = (st["count_top"] / n_top_f) if n_top_f and not np.isnan(n_top_f) else np.nan
        prev_bot = (st["count_bot"] / n_bot_f) if n_bot_f and not np.isnan(n_bot_f) else np.nan
        lift = (prev_top - prev_bot) if (not np.isnan(prev_top) and not np.isnan(prev_bot)) else np.nan

        p_value = np.nan
        if _HAVE_SCIPY and n_top and n_bot:
            a = int(st["count_top"])            # in top16
            b = int(n_top - st["count_top"])   # not in top16
            c = int(st["count_bot"])           # in bottom-half
            d = int(n_bot - st["count_bot"])   # not in bottom-half
            try:
                _, p_value = fisher_exact([[a, b], [c, d]], alternative="two-sided")
            except Exception:
                p_value = np.nan

        rows.append(
            {
                "card": name,
                "decks_with_card": int(st["count_all"]),
                "prevalence_all": prev_all,
                "prevalence_top16": prev_top,
                "prevalence_bottom_half": prev_bot,
                "lift_top16_minus_bottom": lift,
                "p_value_fisher": p_value,
            }
        )

    out = pd.DataFrame(rows)
    out = out.dropna(subset=["lift_top16_minus_bottom"], how="any")
    out = out.sort_values(["lift_top16_minus_bottom", "prevalence_all"], ascending=[False, False])
    return out

# -----------------------------
# Best-deck builder + Moxfield export
# -----------------------------

def build_best_deck(perf_df: pd.DataFrame, size: int, min_prev: float) -> pd.DataFrame:
    """Return a ranked decklist of the top-N strongest cards by lift, with a prevalence floor."""
    if perf_df.empty:
        return pd.DataFrame(columns=["rank", "card", "lift", "prevalence_top16", "prevalence_bottom_half", "decks_with_card"])
    df = perf_df.copy()
    df = df[df["prevalence_all"] >= float(min_prev)].copy()
    df = df.sort_values(["lift_top16_minus_bottom", "prevalence_all"], ascending=[False, False])
    df = df.head(int(size)).copy()
    df.insert(0, "rank", range(1, len(df) + 1))
    df.rename(columns={
        "lift_top16_minus_bottom": "lift",
    }, inplace=True)
    return df[["rank", "card", "lift", "prevalence_top16", "prevalence_bottom_half", "decks_with_card"]]


def write_moxfield_list(path: str, cards: List[str], commander_names: Optional[str] = None) -> None:
    """Write a paste-ready Moxfield list with optional Commander section.

    Format accepted by Moxfield (Import > Paste):
    Commander\n1 Commander Name\n[1 Partner Commander]\n\nDeck\n1 Card A\n1 Card B\n...
    """
    lines: List[str] = []
    if commander_names:
        lines.append("Commander")
        for name in commander_names.split(" / "):
            name = name.strip()
            if name:
                lines.append(f"1 {name}")
        lines.append("")
    lines.append("Deck")
    for c in cards:
        lines.append(f"1 {c}")
    txt = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)

# -----------------------------
# CLI / main
# -----------------------------

def sanitize_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:120]


def select_commander(by_cmd: pd.DataFrame, target: Optional[str]) -> str:
    if target:
        return target
    print("Top 10 commanders by conversion (recent window):")
    view = by_cmd.head(10).reset_index(drop=True)
    for i, row in view.iterrows():
        conv = row.get("conversion_rate")
        conv_str = f"{conv:.2%}" if pd.notna(conv) else "NA"
        print(f"[{i+1}] {row['commander']} — conv={conv_str} (n={int(row['entries'])})")
    while True:
        choice = input("Pick a commander [1-10]: ").strip()
        if choice.isdigit() and (1 <= int(choice) <= min(10, len(view))):
            idx = int(choice) - 1
            return str(view.loc[idx, "commander"])  # type: ignore
        print("Invalid choice. Try again.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=3, help="How many months back to analyze (approx 30*months days)")
    ap.add_argument("--participant-min", type=int, default=60, help="Minimum participants per tournament")
    ap.add_argument("--api-key", type=str, default=None, help="TopDeck.gg API key (or env TOPDECK_API_KEY)")
    ap.add_argument("--commander", type=str, default=None, help="Pick a specific commander (skip interactive prompt)")
    ap.add_argument("--out-dir", type=str, default=".", help="Directory to write CSV outputs")
    ap.add_argument("--min-deck-entries", type=int, default=None, help="Minimum total entries per commander over the window (default: 5 × months)")
    ap.add_argument("--require-participants", action="store_true", help="Only include rows where tournament participants count is known (>0)")
    # Best-deck dials
    ap.add_argument("--best-deck", action="store_true", help="Build theoretical best deck from selected commander's strongest cards")
    ap.add_argument("--best-deck-global", action="store_true", help="Build theoretical best deck from all eligible commanders combined")
    ap.add_argument("--best-deck-size", type=int, default=99, help="Number of non-commander cards to include (default 99)")
    ap.add_argument("--best-deck-min-prev", type=float, default=0.15, help="Minimum overall prevalence for a card to be eligible (0..1; default 0.15)")
    args = ap.parse_args()

    api_key = args.api_key or os.getenv("TOPDECK_API_KEY")
    if not api_key:
        raise SystemExit("Provide API key with --api-key or set TOPDECK_API_KEY environment variable.")

    last_days = int(30 * args.months)
    print(f"Fetching tournaments for the last {last_days} days (participant_min={args.participant_min})…")
    df = build_rows(api_key, last_days=last_days, participant_min=args.participant_min)
    if df.empty:
        raise SystemExit("No standings/deck data found. Check API key/permissions or adjust filters.")

    # Optional: require participants known
    if args.require_participants:
        before = len(df)
        df = df[df["participants"].notna() & (df["participants"] > 0)].copy()
        print(f"Filtered rows with known participants: {before} -> {len(df)}")

    # Apply commander eligibility threshold: at least M entries over the window
    min_required = args.min_deck_entries if args.min_deck_entries is not None else 5 * args.months
    counts = df.groupby("commander").size().rename("entries_total").reset_index()
    eligible_cmds = counts[counts["entries_total"] >= int(min_required)]["commander"]
    df = df[df["commander"].isin(eligible_cmds)].copy()
    if df.empty:
        raise SystemExit(
            f"No commanders meet the minimum entries threshold (min_deck_entries={min_required}). "
            "Lower --min-deck-entries or extend the window."
        )

    # Compute commanders by conversion across last ~N months (in weeks)
    approx_weeks = max(1, round(last_days / 7))
    by_cmd = top_commanders_by_conversion(df, weeks_back=approx_weeks)
    if by_cmd.empty:
        raise SystemExit("No commander stats available in the selected window.")

    # If we're doing a GLOBAL best-deck, we can skip picking a commander
    commander = None
    if not args.best_deck_global:
        commander = select_commander(by_cmd, args.commander)
        print(f"Selected commander: {commander}")

    # Scope subset (commander vs global)
    df_scope = df[df["commander"] == commander].copy() if commander else df.copy()
    if df_scope.empty:
        raise SystemExit("No decks found for the selected scope.")

    # Classify deck performance
    df_scope = classify_deck_rows(df_scope)

    # Persist deck appearances for the *selected commander* only (if one was chosen)
    os.makedirs(args.out_dir, exist_ok=True)
    if commander:
        decks_csv = os.path.join(args.out_dir, f"decks_{sanitize_filename(commander)}.csv")
        df_scope.to_csv(decks_csv, index=False)
        print(f"Wrote {decks_csv} ({len(df_scope)} rows)")

    # Compute per-card performance and write CSV
    perf = compute_card_performance(df_scope)
    scope_tag = sanitize_filename(commander) if commander else "GLOBAL"
    perf_csv = os.path.join(args.out_dir, f"card_performance_{scope_tag}.csv")
    perf.to_csv(perf_csv, index=False)
    print(f"Wrote {perf_csv} ({len(perf)} rows)")

    # Optional: build theoretical best deck (Moxfield paste)
    if args.best_deck or args.best_deck_global:
        best = build_best_deck(perf, size=args.best_deck_size, min_prev=args.best_deck_min_prev)
        best_csv = os.path.join(args.out_dir, f"best_deck_{scope_tag}.csv")
        best.to_csv(best_csv, index=False)
        # Moxfield text (with Commander section if single-commander scope)
        mox_txt = os.path.join(args.out_dir, f"best_deck_{scope_tag}_moxfield.txt")
        write_moxfield_list(mox_txt, best["card"].tolist(), commander_names=commander)
        print(f"Wrote {best_csv} and {mox_txt} — paste the TXT into Moxfield (Import > Paste)")

    # Pretty print quick summary
    def preview(df_: pd.DataFrame, title: str, head: bool = True):
        print("\n" + title)
        print("=" * len(title))
        view = df_.head(20) if head else df_.tail(20)
        with pd.option_context("display.max_rows", 50, "display.max_columns", 10, "display.width", 140):
            cols = [
                "card",
                "decks_with_card",
                "prevalence_all",
                "prevalence_top16",
                "prevalence_bottom_half",
                "lift_top16_minus_bottom",
                "p_value_fisher",
            ]
            cols = [c for c in cols if c in view.columns]
            print(view[cols])

    preview(perf, title="Top 20 cards (highest lift: Top16 vs Bottom-half)")
    preview(perf.sort_values("lift_top16_minus_bottom", ascending=True), title="Bottom 20 cards (most negative lift)")


if __name__ == "__main__":
    main()
