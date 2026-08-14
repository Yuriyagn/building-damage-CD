from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from .datasets import Stage2DamageDataset


def build_derangement(rows: list[dict[str, Any]], mode: str, seed: int) -> tuple[list[int], dict[str, Any]]:
    """Return target-index -> SAR-source-index mapping without self-pairs."""
    normalized = mode.lower().replace("-", "_")
    if normalized in {"none", "paired", "identity"}:
        mapping = list(range(len(rows)))
        return mapping, {"mode": "paired", "seed": seed, "singleton_groups": [], "records": []}

    groups: dict[str, list[int]] = defaultdict(list)
    if normalized in {"global", "across_event", "across_events"}:
        groups["__global__"] = list(range(len(rows)))
        normalized = "global"
    elif normalized in {"within_event", "event"}:
        for index, row in enumerate(rows):
            groups[str(row.get("event_id", "") or "unknown")].append(index)
        normalized = "within_event"
    else:
        raise ValueError(f"unknown SAR shuffle mode: {mode}")

    rng = random.Random(seed)
    mapping = list(range(len(rows)))
    singleton_groups: list[str] = []
    records: list[dict[str, Any]] = []
    for group_name in sorted(groups):
        indices = list(groups[group_name])
        if len(indices) < 2:
            singleton_groups.append(group_name)
            continue
        order = list(indices)
        rng.shuffle(order)
        shift = rng.randint(1, len(order) - 1)
        sources = order[shift:] + order[:shift]
        for target_index, source_index in zip(order, sources, strict=True):
            if target_index == source_index:
                raise AssertionError("derangement generated a self-pair")
            mapping[target_index] = source_index
            records.append(
                {
                    "group": group_name,
                    "target_index": target_index,
                    "target_id": str(rows[target_index].get("id", "")),
                    "source_index": source_index,
                    "source_id": str(rows[source_index].get("id", "")),
                }
            )

    if singleton_groups:
        raise ValueError(
            f"cannot create {normalized} derangement; singleton groups: {singleton_groups[:20]}"
        )
    return mapping, {
        "mode": normalized,
        "seed": seed,
        "singleton_groups": singleton_groups,
        "records": sorted(records, key=lambda row: int(row["target_index"])),
    }


class Stage2V2Dataset(Stage2DamageDataset):
    """Stage-2 v2 dataset with a fixed, auditable SAR permutation."""

    def __init__(
        self,
        *args: Any,
        sar_shuffle_mode: str = "paired",
        sar_shuffle_seed: int = 42,
        sar_singleton_policy: str = "error",
        cache_items: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        original_rows = [dict(row) for row in self.rows]
        excluded_rows: list[dict[str, str]] = []
        normalized_mode = sar_shuffle_mode.lower().replace("-", "_")
        if normalized_mode in {"within_event", "event"} and sar_singleton_policy == "exclude":
            counts: dict[str, int] = defaultdict(int)
            for row in original_rows:
                counts[str(row.get("event_id", "") or "unknown")] += 1
            kept_rows = []
            for row in original_rows:
                event_id = str(row.get("event_id", "") or "unknown")
                if counts[event_id] < 2:
                    excluded_rows.append({"id": str(row.get("id", "")), "event_id": event_id})
                else:
                    kept_rows.append(row)
            original_rows = kept_rows
        elif sar_singleton_policy not in {"error", "exclude"}:
            raise ValueError(f"unknown SAR singleton policy: {sar_singleton_policy}")
        mapping, info = build_derangement(original_rows, sar_shuffle_mode, sar_shuffle_seed)
        info["singleton_policy"] = sar_singleton_policy
        info["excluded_singletons"] = excluded_rows
        info["excluded_singleton_count"] = len(excluded_rows)
        self.sar_permutation = mapping
        self.permutation_info = info
        self.cache_items = bool(cache_items)
        self._item_cache: dict[int | tuple[int, int], dict[str, Any]] = {}
        self._sar_source_ids: list[str] = []
        remapped_rows: list[dict[str, Any]] = []
        for target_index, source_index in enumerate(mapping):
            target = dict(original_rows[target_index])
            source = original_rows[source_index]
            target["post_sar"] = source["post_sar"]
            self._sar_source_ids.append(str(source.get("id", "")))
            remapped_rows.append(target)
        self.rows = remapped_rows

    def __getitem__(self, index: int | tuple[int, int]) -> dict[str, Any]:
        if index in self._item_cache:
            return self._item_cache[index]
        item = super().__getitem__(index)
        row_index = int(index[0]) if isinstance(index, tuple) else int(index)
        item["sar_source_id"] = self._sar_source_ids[row_index]
        item["sar_is_paired"] = bool(self.sar_permutation[row_index] == row_index)
        if self.cache_items:
            self._item_cache[index] = item
        return item
