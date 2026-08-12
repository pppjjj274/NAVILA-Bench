#!/usr/bin/env python3
"""Export annotated 8-frame RGB contact sheets from a Safe-VLN episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont, ImageOps

from safe_vln.actions import ACTIONS
from safe_vln.dataset import iter_sample_refs, load_sample_refs


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--only-recovery", action="store_true")
    parser.add_argument("--frame-width", type=int, default=320)
    return parser.parse_args()


def action_text(action_id):
    if isinstance(action_id, int) and 0 <= action_id < len(ACTIONS):
        return ACTIONS[action_id].text
    return "unknown"


def number(value, digits=3):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def probability(metadata, action_id):
    values = metadata.get("action_probabilities")
    if isinstance(values, list) and isinstance(action_id, int):
        if 0 <= action_id < len(values):
            return number(values[action_id], 4)
    return "n/a"


def _line(draw, xy, text, *, fill, font, width):
    x, y = xy
    for row in textwrap.wrap(str(text), width=width) or [""]:
        draw.text((x, y), row, fill=fill, font=font)
        y += 15
    return y


def render_contact_sheet(frames, metadata, frame_width):
    if len(frames) != 8:
        raise ValueError("Safe-VLN visualizer requires exactly eight frames")
    if frame_width < 64:
        raise ValueError("frame width must be at least 64")
    frame_height = round(frame_width * 9 / 16)
    font = ImageFont.load_default()
    recovery = str(metadata.get("recovery_category", "unlabeled"))
    border = {
        "forward_after_turn": "#d62728",
        "action_mismatch": "#ff7f0e",
        "action_match": "#2ca02c",
    }.get(recovery, "#777777")
    tile_height = frame_height + 18
    panel = Image.new("RGB", (frame_width * 4, tile_height * 2), "black")
    draw = ImageDraw.Draw(panel)
    alignment = metadata.get("frame_alignment")
    if not isinstance(alignment, list):
        alignment = [{} for _ in frames]
    for index, frame in enumerate(frames):
        tile = ImageOps.pad(
            frame.convert("RGB"), (frame_width, frame_height), color="black"
        )
        x = (index % 4) * frame_width
        y = (index // 4) * tile_height
        panel.paste(tile, (x, y))
        draw.rectangle((x, y, x + frame_width - 1, y + frame_height - 1), outline=border, width=3)
        padding = (
            index < len(alignment)
            and bool(alignment[index].get("history_padding", False))
        )
        label = f"history frame {index + 1}/8"
        if padding:
            label += " (repeated first)"
        draw.text(
            (x + 4, y + frame_height + 2),
            label,
            fill="#b0b0b0" if padding else "white",
            font=font,
        )

    oracle_id = metadata.get("oracle_action_id")
    policy_id = metadata.get("policy_action_id")
    executed_id = metadata.get("executed_action_id")
    footer = Image.new("RGB", (panel.width, 142), "#151515")
    footer_draw = ImageDraw.Draw(footer)
    y = 5
    y = _line(
        footer_draw,
        (8, y),
        f"episode={metadata.get('episode_id')} state={metadata.get('index')} "
        f"recovery={recovery} turn_streak={metadata.get('consecutive_turn_actions', 0)}",
        fill=border,
        font=font,
        width=155,
    )
    y = _line(
        footer_draw,
        (8, y),
        "instruction: " + str(metadata.get("instruction", "unknown")),
        fill="white",
        font=font,
        width=155,
    )
    y = _line(
        footer_draw,
        (8, y),
        f"oracle: {action_text(oracle_id)} [p={probability(metadata, oracle_id)}] | "
        f"policy: {action_text(policy_id)} [p={probability(metadata, policy_id)}] | "
        f"executed: {action_text(executed_id)}",
        fill="white",
        font=font,
        width=155,
    )
    _line(
        footer_draw,
        (8, y),
        f"reward={number(metadata.get('reward'))} cost={number(metadata.get('cost'))} "
        f"distance={number(metadata.get('distance_after'))} "
        f"hard={bool(metadata.get('hard_violation', False))} "
        f"done={bool(metadata.get('done', False))}",
        fill="white",
        font=font,
        width=155,
    )
    result = Image.new("RGB", (panel.width, panel.height + footer.height), "black")
    result.paste(panel, (0, 0))
    result.paste(footer, (0, panel.height))
    return result


def main():
    args = parse_args()
    refs = [
        ref
        for ref in iter_sample_refs(args.dataset_dir, args.split)
        if str(ref.metadata.get("episode_id")) == str(args.episode_id)
        and (
            not args.only_recovery
            or ref.metadata.get("recovery_category") == "forward_after_turn"
        )
    ]
    refs.sort(key=lambda ref: int(ref.metadata.get("index", -1)))
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        refs = refs[: args.limit]
    if not refs:
        raise RuntimeError("no matching Safe-VLN samples found")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = []
    for frames, metadata in load_sample_refs(refs):
        index = int(metadata.get("index", len(exported)))
        path = output_dir / f"state_{index:06d}.jpg"
        render_contact_sheet(frames, metadata, args.frame_width).save(
            path, quality=92
        )
        exported.append({"index": index, "path": path.name, "metadata": metadata})
    (output_dir / "index.json").write_text(
        json.dumps(exported, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"episode_id": args.episode_id, "sheets": len(exported), "output_dir": str(output_dir)}))


if __name__ == "__main__":
    main()
