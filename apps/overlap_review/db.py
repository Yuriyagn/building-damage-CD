from __future__ import annotations

import csv
import os
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


REVIEW_LABELS = {
    "confirmed_overlap",
    "likely_overlap",
    "adjacent_no_overlap",
    "same_region_no_overlap",
    "not_overlap",
    "uncertain",
}
REVIEW_ACTIONS = {
    "exclude_train_side",
    "exclude_test_side",
    "keep_both_record_only",
    "cluster_same_area",
    "needs_second_review",
}


class CandidateStore:
    def __init__(self, audit_root: Path):
        self.audit_root = audit_root.resolve()
        self.candidates_path = self.audit_root / "candidates_all.csv"
        self.decisions_path = self.audit_root / "review_decisions.csv"
        if not self.candidates_path.is_file():
            raise FileNotFoundError(f"missing candidates: {self.candidates_path}")
        self.lock = threading.Lock()
        self.candidates = self._read_csv(self.candidates_path)
        self.by_id = {row["pair_id"]: row for row in self.candidates}
        self.decisions = {
            row["pair_id"]: row
            for row in self._read_csv(self.decisions_path)
            if row.get("pair_id")
        }

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, str]]:
        if not path.is_file():
            return []
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def merged(self, pair_id: str) -> dict[str, str]:
        if pair_id not in self.by_id:
            raise KeyError(pair_id)
        row = dict(self.by_id[pair_id])
        decision = self.decisions.get(pair_id, {})
        row.update({key: value for key, value in decision.items() if value})
        row["review_status"] = "reviewed" if decision.get("review_label") else "unreviewed"
        return row

    def query(
        self,
        split_pair: str | None,
        risk: str | None,
        status: str | None,
        offset: int,
        limit: int,
    ) -> dict[str, object]:
        rows = [self.merged(row["pair_id"]) for row in self.candidates]
        if split_pair:
            rows = [row for row in rows if row["split_pair"] == split_pair]
        if risk:
            wanted = set(risk.split(","))
            rows = [row for row in rows if row["risk_level"] in wanted]
        if status:
            rows = [row for row in rows if row["review_status"] == status]
        total = len(rows)
        return {"total": total, "offset": offset, "limit": limit, "items": rows[offset : offset + limit]}

    def summary(self) -> dict[str, object]:
        merged = [self.merged(row["pair_id"]) for row in self.candidates]
        mandatory = [
            row
            for row in merged
            if row["split_pair"] == "train-test" and row["risk_level"] in {"high", "medium"}
        ]
        return {
            "audit_root": str(self.audit_root),
            "total": len(merged),
            "risk": dict(Counter(row["risk_level"] for row in merged)),
            "split_pair": dict(Counter(row["split_pair"] for row in merged)),
            "review_status": dict(Counter(row["review_status"] for row in merged)),
            "review_label": dict(Counter(row.get("review_label", "") for row in merged if row.get("review_label"))),
            "risk_review_status": {
                risk: dict(Counter(row["review_status"] for row in merged if row["risk_level"] == risk))
                for risk in ("high", "medium", "low")
            },
            "mandatory_train_test": {
                "total": len(mandatory),
                "reviewed": sum(row["review_status"] == "reviewed" for row in mandatory),
                "unreviewed": sum(row["review_status"] == "unreviewed" for row in mandatory),
            },
        }

    def save_decision(self, pair_id: str, label: str, action: str, comment: str) -> dict[str, str]:
        if pair_id not in self.by_id:
            raise KeyError(pair_id)
        if label not in REVIEW_LABELS:
            raise ValueError(f"invalid review_label: {label}")
        if action not in REVIEW_ACTIONS:
            raise ValueError(f"invalid review_action: {action}")
        decision = {
            "pair_id": pair_id,
            "review_label": label,
            "review_action": action,
            "review_comment": comment.strip(),
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
        }
        with self.lock:
            self.decisions[pair_id] = decision
            self._flush()
        return self.merged(pair_id)

    def _flush(self) -> None:
        fields = ["pair_id", "review_label", "review_action", "review_comment", "reviewed_at"]
        temp = self.decisions_path.with_name(f".{self.decisions_path.name}.{os.getpid()}.tmp")
        with temp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.decisions[key] for key in sorted(self.decisions))
        os.replace(temp, self.decisions_path)
