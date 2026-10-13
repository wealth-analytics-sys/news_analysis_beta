"""Load the metrics and entity dictionary seeds into the database.

    python -m src.load_dictionaries            # upsert seeds
    python -m src.load_dictionaries --dry-run  # show what would be written

Seeds live in dictionaries/*.yaml and are AWAITING RUSHABH'S REVIEW
(CLAUDE.md §9 open question 4). `new_metric` / `new_entity` proposals coming out of S4 are
surfaced for review and are never auto-added here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from . import db
from .config import REPO_ROOT, load_config

DICT_DIR = REPO_ROOT / "dictionaries"

METRIC_SQL = """
INSERT INTO metrics_dictionary (canonical_name, aliases, category, default_persistence_tier, notes)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (canonical_name) DO UPDATE SET
  aliases = EXCLUDED.aliases,
  category = EXCLUDED.category,
  default_persistence_tier = EXCLUDED.default_persistence_tier
"""

ENTITY_SQL = """
INSERT INTO entity_dictionary (canonical_slug, aliases, notes)
VALUES (%s, %s, %s)
ON CONFLICT (canonical_slug) DO UPDATE SET aliases = EXCLUDED.aliases
"""


def load_seeds() -> tuple[list[dict], list[dict]]:
    metrics = yaml.safe_load((DICT_DIR / "metrics_seed.yaml").read_text(encoding="utf-8"))["metrics"]
    entities = yaml.safe_load((DICT_DIR / "entities_seed.yaml").read_text(encoding="utf-8"))["entities"]
    return metrics, entities


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    metrics, entities = load_seeds()
    tiers: dict[str, int] = {}
    for m in metrics:
        tiers[m["default_persistence_tier"]] = tiers.get(m["default_persistence_tier"], 0) + 1

    print(f"metrics  : {len(metrics)}  by tier: {tiers}")
    print(f"entities : {len(entities)}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    config = load_config()
    db.execute_many(config, METRIC_SQL, [
        (m["canonical_name"], m.get("aliases", []), m.get("category"),
         m.get("default_persistence_tier"), m.get("notes"))
        for m in metrics
    ])
    db.execute_many(config, ENTITY_SQL, [
        (e["canonical_slug"], e.get("aliases", []), e.get("notes")) for e in entities
    ])
    print("\nseeded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
