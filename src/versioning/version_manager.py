import os
import json
import numpy as np
from pathlib import Path

class VersionManager:
    def __init__(self, base_dir: str = 'data/indexes'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.base_dir / 'version_metadata.json'
        self.versions = {}
        if self.metadata_file.exists():
            self.load_metadata(str(self.metadata_file))

    def register_version(self, version: str, metadata: dict):
        self.versions[version] = metadata
        self.save_metadata(str(self.metadata_file))

    def get_version_metadata(self, version: str) -> dict:
        return self.versions.get(version, {})

    def list_versions(self) -> list[str]:
        return list(self.versions.keys())

    def filter_results_by_version(self, results: list[dict], version: str) -> list[dict]:
        return [r for r in results if r.get('version') == version]

    def detect_near_duplicates(self, embeddings: np.ndarray, threshold: float = 0.95) -> list[tuple]:
        sim_matrix = np.dot(embeddings, embeddings.T)
        n = sim_matrix.shape[0]
        duplicates = []
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] >= threshold:
                    duplicates.append((i, j))
        return duplicates

    def deduplicate_results(self, results: list[dict], version_priority: str = 'latest') -> list[dict]:
        seen = {}
        for r in results:
            key = r.get('doc_id', str(id(r)))
            if key not in seen:
                seen[key] = r
            else:
                existing = seen[key]
                v_new = r.get('version', '')
                v_old = existing.get('version', '')
                if version_priority == 'latest' and v_new > v_old:
                    seen[key] = r
                elif version_priority != 'latest' and v_new == version_priority:
                    seen[key] = r
        return list(seen.values())

    def save_metadata(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.versions, f, indent=2)

    def load_metadata(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            self.versions = json.load(f)
