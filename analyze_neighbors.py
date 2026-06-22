#!/usr/bin/env python
"""
Analyze nearest-neighbor results: for each player, report which *other*
players appear in their top-5 (appearance and mask models separately).
"""

import os
import re
import argparse
from collections import defaultdict


def parse_results_txt(path):
    """
    Parse a results.txt and return query_player, app_neighbors, mask_neighbors.
    Each neighbor list contains (player, video_name, dist) tuples.
    """
    query_player = None
    app_neighbors = []
    mask_neighbors = []

    section = None  # "app" or "mask"
    rank_re = re.compile(
        r"^\s*Rank\s+\d+:\s*(.+?)\s*/\s*(\S+)\s+\(dist=([\d.]+)\)"
    )

    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n")

            if line.startswith("Query Player"):
                query_player = line.split(":", 1)[1].strip()
            elif "Appearance Model" in line:
                section = "app"
            elif "Mask Model" in line:
                section = "mask"
            else:
                m = rank_re.match(line)
                if m:
                    player, video, dist = m.group(1), m.group(2), float(m.group(3))
                    entry = (player, video, dist)
                    if section == "app":
                        app_neighbors.append(entry)
                    elif section == "mask":
                        mask_neighbors.append(entry)

    return query_player, app_neighbors, mask_neighbors


def main():
    parser = argparse.ArgumentParser(
        description="Analyze nearest-neighbor results across all players"
    )
    parser.add_argument(
        "--results-dir", type=str,
        default="/fs/scratch/PAS3184/v3_vis/nearest_neighbor",
        help="root directory containing per-player result folders",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="optional path to save the report (prints to stdout if omitted)",
    )
    args = parser.parse_args()

    player_dirs = sorted(
        d for d in os.listdir(args.results_dir)
        if os.path.isdir(os.path.join(args.results_dir, d))
    )

    # Per-player stats
    per_player = {}
    # Global counters: how often each player appears as an "other" neighbor
    app_other_count = defaultdict(int)
    mask_other_count = defaultdict(int)

    total_app_other = 0
    total_mask_other = 0

    lines = []

    def log(msg=""):
        lines.append(msg)

    log("=" * 70)
    log("Nearest-Neighbor Analysis: other-player neighbors in top-5")
    log("=" * 70)

    for pdir in player_dirs:
        txt = os.path.join(args.results_dir, pdir, "results.txt")
        if not os.path.isfile(txt):
            continue

        query_player, app_nb, mask_nb = parse_results_txt(txt)
        if query_player is None:
            continue

        app_others = [(p, v, d) for p, v, d in app_nb if p != query_player]
        mask_others = [(p, v, d) for p, v, d in mask_nb if p != query_player]

        per_player[query_player] = {
            "app_self": len(app_nb) - len(app_others),
            "app_others": app_others,
            "mask_self": len(mask_nb) - len(mask_others),
            "mask_others": mask_others,
        }

        total_app_other += len(app_others)
        total_mask_other += len(mask_others)
        for p, _, _ in app_others:
            app_other_count[p] += 1
        for p, _, _ in mask_others:
            mask_other_count[p] += 1

        # Per-player detail
        log(f"\n{query_player}")
        log(f"  [App]  self={len(app_nb) - len(app_others)}, "
            f"other={len(app_others)}/{len(app_nb)}")
        if app_others:
            for p, v, d in app_others:
                log(f"         → {p} / {v}  (dist={d:.4f})")

        log(f"  [Mask] self={len(mask_nb) - len(mask_others)}, "
            f"other={len(mask_others)}/{len(mask_nb)}")
        if mask_others:
            for p, v, d in mask_others:
                log(f"         → {p} / {v}  (dist={d:.4f})")

    # ── Global summary ────────────────────────────────────────────────────
    n = len(per_player)
    log(f"\n{'=' * 70}")
    log("GLOBAL SUMMARY")
    log(f"{'=' * 70}")
    log(f"Players analysed : {n}")
    log(f"App  – total other-player hits : {total_app_other} / {n * 5}  "
        f"({total_app_other / max(n * 5, 1) * 100:.1f}%)")
    log(f"Mask – total other-player hits : {total_mask_other} / {n * 5}  "
        f"({total_mask_other / max(n * 5, 1) * 100:.1f}%)")

    if app_other_count:
        log(f"\nMost frequent other-player neighbors (Appearance):")
        for p, cnt in sorted(app_other_count.items(), key=lambda x: -x[1])[:20]:
            log(f"  {p:30s}  appeared {cnt} time(s)")

    if mask_other_count:
        log(f"\nMost frequent other-player neighbors (Mask):")
        for p, cnt in sorted(mask_other_count.items(), key=lambda x: -x[1])[:20]:
            log(f"  {p:30s}  appeared {cnt} time(s)")

    report = "\n".join(lines)
    print(report)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(report + "\n")
        print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
