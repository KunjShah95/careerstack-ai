import json
from pathlib import Path

DATA_DIR = Path("./data")


def _collection_dir(collection: str) -> Path:
    path = DATA_DIR / collection
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(collection: str, key: str, data: dict) -> None:
    path = _collection_dir(collection) / f"{key}.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_json(collection: str, key: str) -> dict | None:
    path = _collection_dir(collection) / f"{key}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_keys(collection: str) -> list[str]:
    path = _collection_dir(collection)
    return [p.stem for p in path.glob("*.json")]
