"""Cluster segment descriptors into skills and build the skill library.

No Isaac Sim required:

    python scripts/cluster_skills.py \
        --input outputs/segments_descriptors_ant.pkl \
        --num_skills 8 \
        --output outputs/skill_library_ant.pkl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from skill_discovery.clustering.skill_clusterer import SkillClusterer
from skill_discovery.library.skill_library import SkillLibrary
from skill_discovery.utils.buffers import load_pickle
from skill_discovery.utils.logging import get_logger
from skill_discovery.utils.plotting import (
    plot_descriptor_pca,
    plot_skill_descriptors,
    plot_skill_displacements,
    plot_skill_histogram,
)

logger = get_logger("cluster")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster descriptors into skills.")
    parser.add_argument("--input", type=str, required=True, help="Pickle from extract_descriptors.py.")
    parser.add_argument("--output", type=str, required=True, help="Output skill library pickle.")
    parser.add_argument("--num_skills", type=int, default=8)
    parser.add_argument("--method", type=str, default="kmeans", choices=["kmeans", "gmm", "hdbscan"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plots_dir", type=str, default="outputs/plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_pickle(args.input)
    segments = data["segments"]
    descriptors = data["descriptor_matrix"]
    names = data["descriptor_names"]

    clusterer = SkillClusterer(method=args.method, num_skills=args.num_skills, random_seed=args.seed)
    skill_ids = clusterer.fit(descriptors)
    clusterer.save(str(Path(args.output).with_suffix(".clusterer.pkl")))

    library = SkillLibrary().build(segments, descriptors, skill_ids, data.get("descriptor_dicts"))
    library.metadata = {
        "task": data.get("task"),
        "num_segments": len(segments),
        "clustering_method": args.method,
        "num_skills": len(library.skills),
        "seed": args.seed,
    }
    library.save(args.output)

    plots = Path(args.plots_dir)
    plot_descriptor_pca(descriptors, skill_ids, plots / "skill_pca.png")
    plot_skill_histogram(skill_ids, plots / "skill_histogram.png")
    centers = np.stack([library.get_skill(sid).center_descriptor for sid in library.skill_ids])
    plot_skill_descriptors(centers, names, plots / "skill_descriptors.png")
    plot_skill_displacements(
        {
            sid: {
                "mean_delta_x": library.get_skill(sid).mean_delta_x,
                "mean_delta_y": library.get_skill(sid).mean_delta_y,
                "mean_delta_yaw": library.get_skill(sid).mean_delta_yaw,
            }
            for sid in library.skill_ids
        },
        plots / "skill_displacements.png",
    )
    logger.info("Plots written to %s", plots)

    print(json.dumps({"task": data.get("task"), "num_segments": len(segments),
                      "num_skills": len(library.skills), "skill_summary": library.summary()}, indent=2))


if __name__ == "__main__":
    main()
