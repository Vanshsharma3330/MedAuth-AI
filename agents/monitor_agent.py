from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Optional

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg"}


class CaseMonitorAgent:
    def __init__(
            self,
            watch_dir: str,
            callback: Callable[[str], None],
            processed_log: Optional[str] = None,
    ):
        self.watch_dir = Path(watch_dir)
        self.callback = callback

        if processed_log:
            self.processed_log = Path(processed_log)
        else:
            self.processed_log = self.watch_dir / "processed.1"

        self.seen_files: set[str] = set()

        if self.processed_log.exists():
            with open(self.processed_log, "r", encoding="utf-8") as f:
                for line in f:
                    filename = line.strip()
                    if filename:
                        self.seen_files.add(filename)


    def run(self) -> List[str]:
        if not self.watch_dir.exists():
            return []

        new_detected_paths: List[str] = []

        for entry in sorted(self.watch_dir.iterdir()):
            if not entry.is_file():
                continue
            if entry.name.startswith("."):
                continue
            if entry.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if entry.name in self.seen_files:
                continue

            full_path_str = str(entry.resolve())

            self.callback(full_path_str)

            with open(self.processed_log, "a", encoding="utf-8") as f:
                f.write(f"{entry.name}\n")

            self.seen_files.add(entry.name)
            new_detected_paths.append(full_path_str)

        return new_detected_paths



if __name__ == "__main__":
    agent = CaseMonitorAgent(
        "data/incoming",
        callback=lambda p: print(f"New PA-request document : {p}"),
    )
    found = agent.run()
    print(f"Detected {len(found)} new file(s).")