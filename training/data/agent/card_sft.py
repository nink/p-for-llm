"""Convert manual Agent cards into deterministic progressive SFT events."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


DEFAULT_AGENT_PROMPT = (
    "You edit tiny ASCII files. One line per turn, no markdown.\n"
    "Commands: S path text (search); R path line count (read); "
    "E path line new_text (replace); F reply (finish).\n"
    "Search before guessing a line. Use at most 4 turns. Replace only after "
    "a matching search/read. Use F after a tool result. Errors look like E:code. "
    "Keep replies short."
)


def _trajectory_blocks(trajectory: str) -> tuple[str, list[tuple[str, str | None]]]:
    """Parse Q plus ordered A/O blocks without interpreting command arguments."""
    lines = trajectory.splitlines()
    question_index = next(
        (index for index, line in enumerate(lines) if line.startswith("Q: ")), None
    )
    if question_index is None:
        raise ValueError("trajectory is missing a Q: line")
    question = lines[question_index]
    blocks: list[tuple[str, str | None]] = []
    index = question_index + 1
    while index < len(lines):
        line = lines[index]
        if not line:
            index += 1
            continue
        if not line.startswith("A: "):
            raise ValueError(f"expected A: line, got {line!r}")
        action = line
        index += 1
        output_lines: list[str] = []
        while index < len(lines) and not lines[index].startswith("A: "):
            if lines[index].startswith("O: "):
                if output_lines:
                    raise ValueError("trajectory has multiple O: lines for one action")
                output_lines.append(lines[index][3:])
            elif output_lines and lines[index]:
                output_lines.append(lines[index])
            elif lines[index]:
                raise ValueError(f"unexpected trajectory line {lines[index]!r}")
            index += 1
        blocks.append((action, "\n".join(output_lines) if output_lines else None))
    if not blocks:
        raise ValueError("trajectory has no actions")
    if len(blocks) > 4:
        raise ValueError("trajectory has more than four actions")
    return question, blocks


def card_to_slices(
    card: dict[str, Any], *, prompt: str = DEFAULT_AGENT_PROMPT
) -> Iterator[dict[str, Any]]:
    """Yield one event record per progressive action prefix in a card."""
    card_id = card.get("card_id")
    if not isinstance(card_id, str) or not card_id or not card_id.isascii():
        raise ValueError("card_id must be a non-empty ASCII string")
    trajectory = card.get("trajectory")
    if not isinstance(trajectory, str) or not trajectory.isascii():
        raise ValueError("card trajectory must be ASCII text")
    question, blocks = _trajectory_blocks(trajectory)
    prefix = [
        {"text": prompt, "loss": False},
        {"text": question, "loss": False},
    ]
    history: list[dict[str, Any]] = []
    for slice_index, (action, output) in enumerate(blocks):
        events = [*prefix, *history, {"text": action, "loss": True}]
        yield {
            "card_id": card_id,
            "split_group": card_id,
            "slice_index": slice_index,
            "events": events,
        }
        history.append({"text": action, "loss": False})
        if output is not None:
            history.append({"text": f"O: {output}", "loss": False})


def iter_cards(paths: Iterable[Path]) -> Iterator[dict[str, Any]]:
    """Read JSON arrays or JSONL card files in stable path and record order."""
    for path in sorted((Path(value) for value in paths), key=lambda item: str(item)):
        if path.is_dir():
            yield from iter_cards(sorted(path.glob("*.json")))
            continue
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"card is not an object: {path}:{line_number}")
                    yield value
            continue
        if path.suffix != ".json":
            raise ValueError(f"unsupported card file: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            yield value
        elif isinstance(value, list):
            for card in value:
                if not isinstance(card, dict):
                    raise ValueError(f"card is not an object: {path}")
                yield card
        else:
            raise ValueError(f"card file must contain an object or array: {path}")


def write_card_slices(
    inputs: Iterable[Path], output: Path, *, prompt: str = DEFAULT_AGENT_PROMPT
) -> dict[str, int]:
    """Write deterministic event JSONL and return card/slice counts."""
    output.parent.mkdir(parents=True, exist_ok=True)
    cards = slices = 0
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for card in iter_cards(inputs):
            cards += 1
            for record in card_to_slices(card, prompt=prompt):
                handle.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")))
                handle.write("\n")
                slices += 1
    return {"cards": cards, "slices": slices}
