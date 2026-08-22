from __future__ import annotations

import random
import json
from collections import defaultdict
from pathlib import Path
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
        sar_permutation_file: str | Path | None = None,
        rq3_reliability_permutation_file: str | Path | None = None,
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
        if sar_permutation_file is not None:
            permutation_path = Path(sar_permutation_file)
            payload = json.loads(permutation_path.read_text(encoding="utf-8"))
            raw_mapping = payload.get("mapping", payload)
            if not isinstance(raw_mapping, dict):
                raise ValueError("SAR permutation file must contain an ID-to-ID mapping")
            id_to_index = {str(row.get("id", "")): index for index, row in enumerate(original_rows)}
            if set(raw_mapping) != set(id_to_index):
                missing = sorted(set(id_to_index) - set(raw_mapping))
                extra = sorted(set(raw_mapping) - set(id_to_index))
                raise ValueError(f"SAR permutation ID mismatch: missing={missing[:5]}, extra={extra[:5]}")
            source_ids = [str(raw_mapping[str(row["id"])]) for row in original_rows]
            if any(source_id not in id_to_index for source_id in source_ids):
                raise ValueError("SAR permutation references an unknown source ID")
            mapping = [id_to_index[source_id] for source_id in source_ids]
            if len(set(mapping)) != len(mapping):
                raise ValueError("SAR permutation must be bijective")
            if any(target == source for target, source in enumerate(mapping)):
                raise ValueError("SAR permutation must be fully deranged")
            info = {
                "mode": "external_fixed",
                "seed": sar_shuffle_seed,
                "source": str(permutation_path.resolve()),
                "records": [
                    {"target_id": str(original_rows[target]["id"]), "source_id": str(original_rows[source]["id"])}
                    for target, source in enumerate(mapping)
                ],
                "singleton_groups": [],
            }
        else:
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
            for key in (
                "sar_provenance",
                "sar_provider",
                "sar_platform",
                "sar_mode",
                "sar_polarization",
                "sar_product_type",
                "sar_acquisition_time",
                "sar_metadata_confidence",
                "metadata_confidence",
                "sar_incidence_angle",
                "sar_incidence_angle_deg",
                "sar_gsd",
                "sar_gsd_m",
            ):
                if key in source:
                    target[key] = source[key]
                else:
                    target.pop(key, None)
            self._sar_source_ids.append(str(source.get("id", "")))
            remapped_rows.append(target)
        self.rows = remapped_rows
        if rq3_reliability_permutation_file is not None:
            reliability_path = Path(rq3_reliability_permutation_file)
            payload = json.loads(reliability_path.read_text(encoding="utf-8"))
            raw_mapping = payload.get("mapping", payload)
            if not isinstance(raw_mapping, dict):
                raise ValueError("RQ3 reliability permutation must contain an ID-to-ID mapping")
            id_to_row = {str(row["id"]): row for row in original_rows}
            if set(raw_mapping) != set(id_to_row):
                raise ValueError("RQ3 reliability permutation IDs do not match the training manifest")
            mixed_rows: list[dict[str, Any]] = []
            mixed_source_ids: list[str] = []
            mixed_mapping: list[int] = []
            metadata_keys = (
                "sar_provenance", "sar_provider", "sar_platform", "sar_mode",
                "sar_polarization", "sar_product_type", "sar_acquisition_time",
                "sar_metadata_confidence", "sar_incidence_angle", "sar_gsd",
                "metadata_confidence", "sar_incidence_angle_deg", "sar_gsd_m",
            )
            for target_index, paired in enumerate(original_rows):
                target_id = str(paired["id"])
                source_id = str(raw_mapping[target_id])
                if source_id == target_id or source_id not in id_to_row:
                    raise ValueError("RQ3 reliability permutation must be fully deranged")
                source = id_to_row[source_id]
                deranged = dict(paired)
                deranged["post_sar"] = source["post_sar"]
                for key in metadata_keys:
                    if key in source:
                        deranged[key] = source[key]
                    else:
                        deranged.pop(key, None)
                mixed_rows.extend((dict(paired), deranged))
                mixed_source_ids.extend((target_id, source_id))
                mixed_mapping.extend((2 * target_index, -1))
            self.rows = mixed_rows
            self._sar_source_ids = mixed_source_ids
            self.sar_permutation = mixed_mapping
            self.permutation_info = {
                "mode": "rq3_reliability_balanced_paired_deranged",
                "source": str(reliability_path.resolve()),
                "paired_count": len(original_rows),
                "deranged_count": len(original_rows),
                "records": [],
                "singleton_groups": [],
                "singleton_policy": sar_singleton_policy,
                "excluded_singletons": [],
                "excluded_singleton_count": 0,
            }

    def __getitem__(self, index: int | tuple[int, int]) -> dict[str, Any]:
        if index in self._item_cache:
            return self._item_cache[index]
        item = super().__getitem__(index)
        row_index = int(index[0]) if isinstance(index, tuple) else int(index)
        item["sar_source_id"] = self._sar_source_ids[row_index]
        if self.permutation_info["mode"] == "rq3_reliability_balanced_paired_deranged":
            item["sar_is_paired"] = bool(row_index % 2 == 0)
        else:
            item["sar_is_paired"] = bool(self.sar_permutation[row_index] == row_index)
        if self.cache_items:
            self._item_cache[index] = item
        return item
