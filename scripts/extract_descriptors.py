"""Segment collected rollouts and extract behavior descriptors.

This stage is pure NumPy — no Isaac Sim required — so it can also run with a
plain python interpreter:

    python scripts/extract_descriptors.py \
        --input outputs/rollouts_ant.pkl \
        --output outputs/segments_descriptors_ant.pkl \
        --segment_horizon 32 --segment_stride 16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skill_discovery.descriptors.locomotion_descriptors import LocomotionDescriptorExtractor
from skill_discovery.segmentation.fixed_horizon_segmenter import FixedHorizonSegmenter
from skill_discovery.utils.buffers import load_pickle, save_pickle
from skill_discovery.utils.logging import get_logger

logger = get_logger("extract")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment rollouts and extract descriptors.")
    parser.add_argument("--input", type=str, required=True, help="Rollouts pickle from collect_exploration.py.")
    parser.add_argument("--output", type=str, required=True, help="Output pickle (segments + descriptors).")
    parser.add_argument("--segment_horizon", type=int, default=32)
    parser.add_argument("--segment_stride", type=int, default=16)
    parser.add_argument("--min_segment_length", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_pickle(args.input)
    trajectories = data["trajectories"]
    logger.info("Loaded %d trajectories from %s", len(trajectories), args.input)

    segmenter = FixedHorizonSegmenter(
        segment_horizon=args.segment_horizon,
        segment_stride=args.segment_stride,
        min_segment_length=args.min_segment_length,
    )
    segments = segmenter.segment_all(trajectories)
    if not segments:
        raise RuntimeError(
            "No segments produced. Episodes may be shorter than min_segment_length; "
            "collect longer rollouts or reduce --min_segment_length."
        )

    extractor = LocomotionDescriptorExtractor()
    descriptor_matrix, descriptor_dicts = extractor.compute_matrix(segments)
    logger.info("Extracted descriptor matrix of shape %s", descriptor_matrix.shape)

    save_pickle(
        {
            "task": data.get("task"),
            "segments": segments,
            "descriptor_matrix": descriptor_matrix,
            "descriptor_dicts": descriptor_dicts,
            "descriptor_names": list(extractor.names),
            "segmentation": vars(segmenter),
        },
        args.output,
    )
    logger.info("Saved %d segments with descriptors to %s", len(segments), args.output)


if __name__ == "__main__":
    main()
