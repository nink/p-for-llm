"""Build a compact ASCII-oriented BPE tokenizer from the pinned Qwen source."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SOURCE_REPOSITORY = "Qwen/Qwen3.5-0.8B-Base"
SOURCE_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
SOURCE_TOKENIZER_SHA256 = (
    "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927"
)
SOURCE_CONFIG_SHA256 = (
    "e611fbccc7c29ef3b1cafb1cb7ea548d189968632901d678fd62be68c47885de"
)

DEFAULT_KEEP_ADDED_TOKENS = (
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|fim_pad|>",
    "<|repo_name|>",
    "<|file_sep|>",
    "<think>",
    "</think>",
)


@dataclass(frozen=True, slots=True)
class PruneConfig:
    target_vocab_size: int = 32_768
    max_seq_len: int = 1_024
    keep_added_tokens: tuple[str, ...] = DEFAULT_KEEP_ADDED_TOKENS

    def __post_init__(self) -> None:
        if self.target_vocab_size <= len(self.keep_added_tokens):
            raise ValueError("target_vocab_size leaves no room for base tokens")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")


@dataclass(frozen=True, slots=True)
class PruneResult:
    source_base_vocab_size: int
    source_merge_count: int
    retained_base_vocab_size: int
    retained_merge_count: int
    retained_added_token_count: int
    target_vocab_size: int
    skipped_non_ascii_merges: int


def _byte_decoder() -> dict[int, int]:
    visible = list(range(ord("!"), ord("~") + 1))
    visible += list(range(ord("¡"), ord("¬") + 1))
    visible += list(range(ord("®"), ord("ÿ") + 1))
    codepoints = visible[:]
    extra = 0
    for byte in range(256):
        if byte not in visible:
            visible.append(byte)
            codepoints.append(256 + extra)
            extra += 1
    return dict(zip(codepoints, visible))


BYTE_DECODER = _byte_decoder()


def decode_token_bytes(token: str) -> bytes:
    try:
        return bytes(BYTE_DECODER[ord(character)] for character in token)
    except KeyError as error:
        raise ValueError(
            f"token contains an unknown byte-level glyph: {token!r}"
        ) from error


def is_ascii_token(token: str, source_id: int) -> bool:
    """Allow byte primitives, then only valid ASCII merged text."""

    if source_id < 256:
        return True
    try:
        decoded = decode_token_bytes(token).decode("utf-8")
    except UnicodeDecodeError:
        return False
    return all(ord(character) < 128 for character in decoded)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_source(path: Path, expected_hash: str, label: str) -> None:
    actual_hash = _sha256(path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"{label} SHA-256 is {actual_hash}, expected {expected_hash}"
        )


def _base_vocab_by_id(vocab: dict[str, int]) -> list[str]:
    ordered = sorted(vocab.items(), key=lambda item: item[1])
    expected = list(range(len(ordered)))
    actual = [token_id for _, token_id in ordered]
    if actual != expected:
        raise ValueError("source base vocabulary IDs must be dense")
    return [token for token, _ in ordered]


def _parse_merge(merge: str | list[str]) -> tuple[str, str]:
    if isinstance(merge, list):
        if len(merge) != 2:
            raise ValueError(f"invalid merge pair: {merge!r}")
        return merge[0], merge[1]
    left, separator, right = merge.partition(" ")
    if not separator or not left or not right:
        raise ValueError(f"invalid merge rule: {merge!r}")
    return left, right


def _merge_text(merge: str | list[str]) -> str:
    left, right = _parse_merge(merge)
    return f"{left} {right}"


def _added_token_definitions(
    tokenizer: dict[str, Any],
    tokenizer_config: dict[str, Any],
    keep: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    definitions: dict[str, dict[str, Any]] = {}
    for record in tokenizer.get("added_tokens", []):
        definitions[record["content"]] = copy.deepcopy(record)
    for token_id, record in tokenizer_config.get("added_tokens_decoder", {}).items():
        content = record["content"]
        if content not in definitions:
            definition = copy.deepcopy(record)
            definition["id"] = int(token_id)
            definitions[content] = definition
    missing = [content for content in keep if content not in definitions]
    if missing:
        raise ValueError(f"required added tokens are missing from source: {missing}")
    return {content: definitions[content] for content in keep}


def _rewrite_tokenizer_config(
    source_config: dict[str, Any],
    added_tokens: list[dict[str, Any]],
    max_seq_len: int,
) -> dict[str, Any]:
    config = copy.deepcopy(source_config)
    config["model_max_length"] = max_seq_len
    config["added_tokens_decoder"] = {
        str(record["id"]): {
            key: value for key, value in record.items() if key != "id"
        }
        for record in added_tokens
    }
    keep_ids = {record["content"] for record in added_tokens}
    config["additional_special_tokens"] = [
        token
        for token in source_config.get("additional_special_tokens", [])
        if token in keep_ids
    ]
    return config


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def prune_qwen_tokenizer(
    source_tokenizer_path: Path,
    source_config_path: Path,
    output_dir: Path,
    config: PruneConfig = PruneConfig(),
) -> PruneResult:
    _verify_source(
        source_tokenizer_path,
        SOURCE_TOKENIZER_SHA256,
        "source tokenizer",
    )
    _verify_source(
        source_config_path,
        SOURCE_CONFIG_SHA256,
        "source tokenizer config",
    )
    source_tokenizer = _load_json(source_tokenizer_path)
    source_config = _load_json(source_config_path)
    model = source_tokenizer.get("model", {})
    source_vocab = model.get("vocab")
    source_merges = model.get("merges")
    if not isinstance(source_vocab, dict) or not isinstance(source_merges, list):
        raise ValueError("source tokenizer must contain a BPE vocab and merges list")
    if model.get("type") != "BPE":
        raise ValueError("only BPE tokenizer sources are supported")

    source_tokens = _base_vocab_by_id(source_vocab)
    if len(source_tokens) < 256:
        raise ValueError("source BPE vocabulary must contain all 256 byte symbols")

    added_definitions = _added_token_definitions(
        source_tokenizer,
        source_config,
        config.keep_added_tokens,
    )
    target_base_size = config.target_vocab_size - len(added_definitions)
    retained_tokens = source_tokens[:256]
    retained_set = set(retained_tokens)
    retained_merges: list[str] = []
    skipped_non_ascii = 0

    for source_merge in source_merges:
        if len(retained_tokens) >= target_base_size:
            break
        left, right = _parse_merge(source_merge)
        merged = left + right
        source_id = source_vocab.get(merged)
        if source_id is None:
            raise ValueError(
                f"merge result missing from source vocabulary: {merged!r}"
            )
        if left not in retained_set or right not in retained_set:
            continue
        if not is_ascii_token(merged, source_id):
            skipped_non_ascii += 1
            continue
        if merged in retained_set:
            continue
        retained_tokens.append(merged)
        retained_set.add(merged)
        retained_merges.append(_merge_text(source_merge))

    if len(retained_tokens) != target_base_size:
        raise ValueError(
            "could not reach target base vocabulary: "
            f"{len(retained_tokens)} of {target_base_size}"
        )

    new_vocab = {token: token_id for token_id, token in enumerate(retained_tokens)}
    new_added_tokens: list[dict[str, Any]] = []
    for offset, content in enumerate(config.keep_added_tokens):
        record = copy.deepcopy(added_definitions[content])
        record["id"] = target_base_size + offset
        new_added_tokens.append(record)

    new_tokenizer = copy.deepcopy(source_tokenizer)
    new_tokenizer["model"]["vocab"] = new_vocab
    new_tokenizer["model"]["merges"] = retained_merges
    new_tokenizer["added_tokens"] = new_added_tokens
    new_config = _rewrite_tokenizer_config(
        source_config,
        new_added_tokens,
        config.max_seq_len,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "tokenizer.json", new_tokenizer)
    _write_json(output_dir / "vocab.json", new_vocab)
    _write_json(output_dir / "tokenizer_config.json", new_config)
    (output_dir / "merges.txt").write_text(
        "\n".join(retained_merges) + "\n",
        encoding="utf-8",
    )

    result = PruneResult(
        source_base_vocab_size=len(source_tokens),
        source_merge_count=len(source_merges),
        retained_base_vocab_size=len(retained_tokens),
        retained_merge_count=len(retained_merges),
        retained_added_token_count=len(new_added_tokens),
        target_vocab_size=config.target_vocab_size,
        skipped_non_ascii_merges=skipped_non_ascii,
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "tokenizer_sha256": SOURCE_TOKENIZER_SHA256,
            "tokenizer_config_sha256": SOURCE_CONFIG_SHA256,
        },
        "policy": {
            "type": "merge-order-ascii-pruning",
            "target_vocab_size": config.target_vocab_size,
            "max_seq_len": config.max_seq_len,
            "keep_added_tokens": list(config.keep_added_tokens),
        },
        "result": asdict(result),
        "files": {},
    }
    for filename in (
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "tokenizer_config.json",
    ):
        manifest["files"][filename] = {
            "size_bytes": (output_dir / filename).stat().st_size,
            "sha256": _sha256(output_dir / filename),
        }
    _write_json(output_dir / "pruning-manifest.json", manifest)
    return result
