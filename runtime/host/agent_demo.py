#!/usr/bin/env python3
"""Run a bounded, in-memory tool-use demonstration on the ESP32-P4."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .p4 import P4Device, ProtocolError, ensure_ready
except ImportError:
    from p4 import P4Device, ProtocolError, ensure_ready


SYSTEM_PROMPT = (
    "You edit tiny ASCII files. One line per turn, no markdown.\n"
    "Commands: S path text (search); R path line count (read); E path line new_text (replace); F reply (finish).\n"
    "Search before guessing a line. Use at most 4 turns. Replace only after a matching search/read. Use F after a tool result.\n"
    "Errors look like E:code. Keep replies short."
)
DEFAULT_TASK = "In services/cache/cache.ini, change cache_policy from lru to lfu."


class VirtualFiles:
    """A deliberately closed file store used by the demo; it never touches disk."""

    def __init__(self) -> None:
        self.files = {
            "services/cache/cache.ini": "[cache]\ncache_policy = lru\nmax_entries = 256\n",
        }
        self.observed: set[tuple[str, int]] = set()

    def _get(self, path: str) -> list[str] | None:
        content = self.files.get(path)
        return None if content is None else content.splitlines()

    def execute(self, command: str) -> str:
        parts = command.strip().split(" ", 2)
        if not parts or not parts[0]:
            return "E:empty_command"
        operation = parts[0]
        if operation == "S":
            if len(parts) != 3 or not parts[1] or not parts[2]:
                return "E:search_syntax"
            lines = self._get(parts[1])
            if lines is None:
                return "E:file_not_found"
            matches = [(number, line) for number, line in enumerate(lines, 1) if parts[2] in line]
            if not matches:
                return "E:text_not_found"
            for number, _line in matches:
                self.observed.add((parts[1], number))
            return "OK\n" + "\n".join(f"{number}: {line}" for number, line in matches)

        if operation == "R":
            fields = command.strip().split()
            if len(fields) != 4 or not fields[1]:
                return "E:read_syntax"
            try:
                first = int(fields[2])
                count = int(fields[3])
            except ValueError:
                return "E:read_numbers"
            lines = self._get(fields[1])
            if lines is None:
                return "E:file_not_found"
            if first < 1 or count < 1 or first > len(lines):
                return "E:line_range"
            last = min(first + count - 1, len(lines))
            for number in range(first, last + 1):
                self.observed.add((fields[1], number))
            return "OK\n" + "\n".join(f"{number}: {lines[number - 1]}" for number in range(first, last + 1))

        if operation == "E":
            fields = command.strip().split(" ", 3)
            if len(fields) != 4 or not fields[1] or not fields[3]:
                return "E:replace_syntax"
            try:
                number = int(fields[2])
            except ValueError:
                return "E:replace_number"
            lines = self._get(fields[1])
            if lines is None:
                return "E:file_not_found"
            if number < 1 or number > len(lines):
                return "E:line_range"
            if (fields[1], number) not in self.observed:
                return "E:line_not_observed"
            if "\n" in fields[3] or "\r" in fields[3]:
                return "E:single_line_only"
            lines[number - 1] = fields[3]
            self.files[fields[1]] = "\n".join(lines) + "\n"
            return f"OK replaced {fields[1]}:{number}"

        return "E:unknown_command"

    def show(self) -> None:
        for path, content in self.files.items():
            print(f"{path}:")
            for number, line in enumerate(content.splitlines(), 1):
                print(f"  {number}: {line}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="USB serial port exposed by the board")
    parser.add_argument("--artifact", type=Path, help="release model artifact used when the board is not loaded")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    files = VirtualFiles()
    with P4Device.connect(args.port, timeout=args.timeout) as device:
        ensure_ready(device, args.artifact)
        device.clear()
        transcript = f"{SYSTEM_PROMPT}\nQ: {args.task}"
        had_tool_result = False
        final_reply = ""
        for turn in range(1, 5):
            device.clear()
            result = device.text(
                transcript + "\n",
                requested_tokens=args.max_new_tokens,
                temperature=0.0,
                top_k=1,
                random_state=1,
            )
            response = result.text.strip()
            command = response.splitlines()[0].strip() if response else ""
            print(f"turn {turn} model: {command or '<empty>'}")
            transcript += f"\n{command}"

            action = command[3:] if command.startswith("A: ") else command

            if action.startswith("F "):
                if not had_tool_result:
                    observation = "E:finish_requires_tool_result"
                    print(f"tool: {observation}")
                    transcript += f"\nO: {observation}"
                    had_tool_result = True
                    continue
                final_reply = action[2:].strip()
                break

            observation = files.execute(action)
            print(f"tool: {observation}")
            transcript += f"\nO: {observation}"
            had_tool_result = True
        else:
            print("agent: turn limit reached")

        if final_reply:
            print(f"final: {final_reply}")
        files.show()


def main() -> int:
    args = parse_args()
    try:
        run(args)
    except (OSError, ProtocolError, RuntimeError, TimeoutError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
