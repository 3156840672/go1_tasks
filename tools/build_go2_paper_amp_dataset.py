"""Build 30 seconds of Go2 Trot plus a short stable standing reference."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


TROT_CATEGORIES = (
  "trot",
  "backward_trot",
  "side_left",
  "side_right",
  "turn_left",
  "turn_right",
)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--source", type=Path, default=Path("logs/amp/go2_reference")
  )
  parser.add_argument(
    "--output", type=Path, default=Path("logs/amp/go2_paper_reference")
  )
  parser.add_argument("--seconds-per-clip", type=float, default=2.5)
  parser.add_argument(
    "--stand-seconds", type=float, default=1.6,
    help="Standing duration; 1.6 / 31.6 seconds gives about 5%% of transitions. "
         "Use 0 to reproduce the original Trot-only dataset in a clean output.",
  )
  parser.add_argument("--overwrite", action="store_true")
  args = parser.parse_args()
  if args.seconds_per_clip <= 0:
    parser.error("--seconds-per-clip must be positive")
  if not np.isfinite(args.stand_seconds) or args.stand_seconds < 0:
    parser.error("--stand-seconds must be finite and nonnegative")

  paths = [
    path
    for category in TROT_CATEGORIES
    for path in sorted((args.source / category).glob("*.npz"))
  ]
  if len(paths) != 12:
    raise ValueError(
      f"Expected 12 directional Trot clips, found {len(paths)} under {args.source}"
    )
  stand_paths = sorted((args.source / "stand").glob("*.npz"))
  if args.stand_seconds > 0:
    if len(stand_paths) != 1:
      raise ValueError(f"Expected one stable stand clip, found {len(stand_paths)}")
    paths += stand_paths
  elif any((args.output / "stand").glob("*.npz")):
    raise ValueError("Trot-only output must not contain existing standing clips")

  total_duration = 0.0
  total_transitions = 0
  moving_duration = 0.0
  standing_transitions = 0
  for source_path in paths:
    with np.load(source_path, allow_pickle=False) as source:
      fps = float(source["fps"])
      frame_count = len(source["root_pos_w"])
      is_stand = source_path.parent.name == "stand"
      seconds = args.stand_seconds if is_stand else args.seconds_per_clip
      keep_frames = round(seconds * fps) + 1
      if keep_frames < 2:
        raise ValueError(f"{source_path}: duration must contain a full transition")
      if keep_frames > frame_count:
        raise ValueError(
          f"{source_path}: needs {keep_frames} frames, only has {frame_count}"
        )
      start = (frame_count - keep_frames) // 2
      stop = start + keep_frames
      output_values = {
        key: (
          value[start:stop]
          if value.ndim > 0 and value.shape[0] == frame_count
          else value
        )
        for key in source.files
        for value in (source[key],)
      }
      if is_stand:
        for field, limit in (("root_lin_vel_b", .02),
                             ("root_ang_vel_b", .02), ("joint_vel", .05)):
          values = output_values[field]
          if not np.isfinite(values).all() or np.sqrt(np.mean(values**2)) > limit:
            raise ValueError(f"{source_path}: unstable standing reference: {field}")

    destination = args.output / source_path.parent.name / source_path.name
    if destination.exists() and not args.overwrite:
      raise FileExistsError(f"Refusing to overwrite {destination}; pass --overwrite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
      destination,
      **output_values,
      source_clip=np.asarray(str(source_path)),
      source_start_frame=np.asarray(start, dtype=np.int32),
      source_stop_frame=np.asarray(stop, dtype=np.int32),
    )
    duration = (keep_frames - 1) / fps
    total_duration += duration
    total_transitions += keep_frames - 1
    if is_stand:
      standing_transitions += keep_frames - 1
    else:
      moving_duration += duration
    print(
      f"{source_path.parent.name:13} {source_path.name:31} "
      f"frames={start}:{stop} duration={duration:.2f}s"
    )

  if not np.isclose(moving_duration, 30.0):
    raise ValueError(f"Trot references must total 30 seconds, got {moving_duration:g}")
  print(
    f"Built {len(paths)} clips, {total_transitions} transitions, "
    f"{total_duration:.1f} seconds under {args.output}"
  )
  print(f"Standing transition share: {standing_transitions / total_transitions:.2%}")


if __name__ == "__main__":
  main()
