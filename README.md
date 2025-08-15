# Commander Card Performance — README

## Overview
This script analyzes competitive EDH (cEDH) tournament data from TopDeck.gg to identify:
- Top commanders by conversion rate to Top 16 finishes.
- Per-card performance within decks for a given commander or across all commanders.
- A theoretical "best deck" constructed from the strongest cards by statistical lift.

It can also export a paste-ready list for [Moxfield.com](https://www.moxfield.com) deck import.

## Features
1. **Tournament Data Retrieval**
   - Fetches recent EDH tournaments from TopDeck.gg.
   - Filters by minimum participants and date range.

2. **Commander Ranking**
   - Calculates conversion rate (Top 16 / total entries) for commanders.
   - Lists the top 10 over the chosen time window.

3. **Per-card Performance Analysis**
   - Computes prevalence and performance lift of each card.
   - Compares Top 16 decks to bottom-half decks.

4. **Theoretical Best Deck**
   - Builds a decklist from the highest-lift cards.
   - Supports commander-specific or global analysis.
   - Outputs both CSV and paste-ready Moxfield text.

5. **Customizable Parameters**
   - Timeframe in months.
   - Minimum deck entries threshold.
   - Minimum participants per tournament.
   - Best deck size and prevalence floor.

## Requirements
- **Python:** 3.9+
- **Libraries:**
  - `requests`
  - `pandas`
  - `numpy`
  - `scipy` *(optional, for p-values)*
- **API Key:** A TopDeck.gg API key with standings and decklist access.

## Installation
```bash
pip install pandas numpy requests
pip install scipy  # optional for p-values
```

## Usage
```bash
python commander_performance.py --months 3 --participant-min 60 --api-key <YOUR_API_KEY>
```

### Options
| Flag | Description |
|------|-------------|
| `--months` | Months back to analyze (default: 3) |
| `--participant-min` | Minimum participants per tournament (default: 60) |
| `--api-key` | TopDeck.gg API key or set `TOPDECK_API_KEY` env var |
| `--commander` | Specific commander name to analyze |
| `--out-dir` | Directory to write CSV outputs |
| `--min-deck-entries` | Minimum total entries per commander over the window |
| `--require-participants` | Only include decks with known participant counts |
| `--best-deck` | Build best deck from selected commander’s cards |
| `--best-deck-size` | Number of cards in best deck (default: 99) Change to 98 for partner pair |

## Output
- `decks_<commander>.csv`: Deck appearances and metadata.
- `card_performance_<scope>.csv`: Per-card performance metrics.
- `best_deck_<scope>.csv`: Ranked list of best-deck cards.
- `best_deck_<scope>_moxfield.txt`: Paste into Moxfield's Import > Paste.

## Example: Build Global Best Deck
```bash
python commander_performance.py --months 6 --participant-min 60 \
  --api-key $TOPDECK_API_KEY --best-deck-global
```

## Notes
- Legality are not enforced for the best-deck output in the case of recent bans.
