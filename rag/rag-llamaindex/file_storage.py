# file_storage.py
import json
import os
from datetime import datetime
from pathlib import Path

class FileStorage:
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.files = {
            "rules":     self.output_dir / "rules.jsonl",
            "warnings":  self.output_dir / "warnings.jsonl",
            "decisions": self.output_dir / "decisions.jsonl",
        }
        self._seen_names = {key: self._load_existing_names(key) for key in self.files}

    def _load_existing_names(self, data_type: str) -> set:
        """Load existing names to avoid duplicates on re-runs."""
        seen = set()
        path = self.files[data_type]
        if path.exists():
            with open(path, "r") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        seen.add(obj.get("name"))
                    except json.JSONDecodeError:
                        continue
        return seen

    def save(self, data_type: str, items: list) -> int:
        """
        Append new items to JSONL file, skipping duplicates.
        Accepts Pydantic models or dicts.
        Returns count of items written.
        """
        if not items or data_type not in self.files:
            return 0

        written = 0
        with open(self.files[data_type], "a", encoding="utf-8") as f:
            for item in items:
                # Handle Pydantic models
                if hasattr(item, "name"):
                    name = item.name
                else:
                    name = item.get("name")
                
                if name in self._seen_names[data_type]:
                    print(f"[SKIP] Duplicate {data_type}: {name}")
                    continue
                item.last_updated = datetime.utcnow().isoformat()

                # Convert to dict for JSON serialization
                if hasattr(item, "model_dump"):
                    item_dict = item.model_dump()
                elif hasattr(item, "dict"):
                    item_dict = item.dict()
                else:
                    item_dict = item
                
                f.write(json.dumps(item_dict, default=str) + "\n")
                self._seen_names[data_type].add(name)
                written += 1

        print(f"[FILE] Saved {written} new {data_type}")
        return written
    
    def load(
        self,
        data_type: str,
        *,
        filters: dict | None = None,
        since: str | None = None,
    ) -> list[dict]:
        """
        Load items from a JSONL file.

        Args:
            data_type: One of "rules", "warnings", "decisions".
            filters:   Optional dict of {field: value} pairs; only items where
                    ALL fields match are returned.  Example:
                        {"severity": "high"}
            since:     Optional ISO-8601 timestamp string.  Only items whose
                    ``last_updated`` field is >= this value are returned.
                    Example: "2025-01-01T00:00:00"

        Returns:
            List of dicts.  Empty list when the file doesn't exist or has no
            matching records.

        Raises:
            ValueError: Unknown data_type.
        """
        if data_type not in self.files:
            raise ValueError(
                f"Unknown data_type {data_type!r}. "
                f"Valid options: {list(self.files)}"
            )

        path = self.files[data_type]
        if not path.exists():
            print(f"[FILE] No file found for {data_type}")
            return []

        results: list[dict] = []

        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj: dict = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"[WARN] Skipping malformed line {lineno} in {data_type}: {exc}")
                    continue

                # --- since filter ---
                if since is not None:
                    last_updated = obj.get("last_updated", "")
                    if last_updated < since:      # ISO strings are lexicographically ordered
                        continue

                # --- field filters (case-insensitive; extracted values like
                # "severity" are not guaranteed to be consistently cased) ---
                if filters and not all(
                    str(obj.get(k, "")).lower() == str(v).lower()
                    for k, v in filters.items()
                ):
                    continue

                results.append(obj)

        print(f"[FILE] Loaded {len(results)} {data_type}")
        return results

    # In file_storage.py — add this method to the class

    def load_all(
        self,
        *,
        filters: dict | None = None,
        since: str | None = None,
        include_type: bool = True,
    ) -> list[dict]:
        """
        Load and merge items from ALL data types into a single list.

        Args:
            filters:      Same field-matching dict as load().
            since:        Same ISO timestamp filter as load().
            include_type: If True, injects a ``_type`` key into every item
                        so you know which file it came from.

        Returns:
            Combined list of dicts, ordered by last_updated ascending.
        """
        combined = []

        for data_type in self.files:
            items = self.load(data_type, filters=filters, since=since)
            if include_type:
                for item in items:
                    item["_type"] = data_type
            combined.extend(items)

        # Sort everything by last_updated so the merged list is chronological
        combined.sort(key=lambda x: x.get("last_updated", ""))

        print(f"[FILE] Loaded {len(combined)} total items across all types")
        return combined