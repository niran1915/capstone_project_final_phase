from pathlib import Path
STORAGE_DIR = Path(__file__).resolve().parents[1] / "storage"
def storage_url(path: Path) -> str:
    rel = Path(path).resolve().relative_to(STORAGE_DIR.resolve()).as_posix()
    return "/storage/" + rel
