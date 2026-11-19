import json
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

BASE_DATA_DIR = Path("data")
CHATS_DIR = BASE_DATA_DIR / "chats"
MEDIA_DIR = BASE_DATA_DIR / "media"

CHATS_DIR.mkdir(parents=True, exist_ok=True)
MEDIA_DIR.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def create_session(title: str = "New Chat") -> Dict:
    session_id = str(uuid.uuid4())[:8]
    now = _now_iso()
    session_data = {
        "id": session_id,
        "title": title,
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }
    save_session(session_data)
    return session_data


def load_session(session_id: str) -> Optional[Dict]:
    path = CHATS_DIR / f"{session_id}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_session(session_data: Dict) -> None:
    session_data["updated_at"] = _now_iso()
    path = CHATS_DIR / f"{session_data['id']}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, ensure_ascii=False, indent=2)


def list_sessions() -> List[Dict]:
    sessions = []
    for file_path in CHATS_DIR.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                sessions.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue
    sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return sessions


def delete_session(session_id: str) -> None:
    path = CHATS_DIR / f"{session_id}.json"
    if path.exists():
        path.unlink()


def save_image_bytes(image_bytes: bytes, prefix: str = "img") -> str:
    """Saves image bytes as PNG and returns the file path."""
    filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:6]}.png"
    file_path = MEDIA_DIR / filename

    img = Image.open(BytesIO(image_bytes))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGBA")
    else:
        img = img.convert("RGB")
    img.save(file_path, format="PNG")
    return str(file_path)