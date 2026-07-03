"""Regenerate skill plots and write a JSON research summary.

No Isaac Sim required:

    python scripts/visualize_skills.py \
        --segments outputs/segments_descriptors_ant.pkl \
        --skill_library outputs/skill_library_ant.pkl

Produces:
    outputs/plots/skill_pca.png
    outputs/plots/skill_histogram.png
    outputs/plots/skill_descriptors.png
    outputs/plots/skill_displacements.png
    outputs/skill_summary.json
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

logger = get_logger("visualize")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize discovered skills.")
    parser.add_argument("--segments", type=str, required=True, help="Pickle from extract_descriptors.py.")
    parser.add_argument("--skill_library", type=str, required=True, help="Pickle from cluster_skills.py.")
    parser.add_argument("--composition_eval", type=str, default=None,
                        help="Optional composition_eval.json to merge into the summary.")
    parser.add_argument("--output_dir", type=str, default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_pickle(args.segments)
    library = SkillLibrary.load(args.skill_library)

    descriptors = data["descriptor_matrix"]
    names = data["descriptor_names"]

    clusterer_path = Path(args.skill_library).with_suffix(".clusterer.pkl")
    if clusterer_path.exists():
        skill_ids = SkillClusterer.load(str(clusterer_path)).predict(descriptors)
    else:
        logger.warning("Clusterer file %s missing; assigning by nearest library center.", clusterer_path)
        centers = np.stack([library.get_skill(sid).center_descriptor for sid in library.skill_ids])
        std = descriptors.std(axis=0) + 1e-8
        dists = np.linalg.norm((descriptors[:, None] - centers[None]) / std, axis=-1)
        skill_ids = np.array(library.skill_ids)[np.argmin(dists, axis=1)]

    plots = Path(args.output_dir) / "plots"
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

    summary = {
        "task": library.metadata.get("task"),
        "num_segments": int(len(descriptors)),
        "num_skills": len(library.skills),
        "skill_summary": library.summary(),
    }
    eval_path = args.composition_eval or str(Path(args.output_dir) / "composition_eval.json")
    if Path(eval_path).exists():
        eval_data = json.loads(Path(eval_path).read_text())
        summary["composition_eval"] = {
            "success_rate": eval_data.get("success_rate"),
            "average_final_distance": eval_data.get("average_final_distance"),
        }

    out = Path(args.output_dir) / "skill_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %s and plots to %s", out, plots)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
