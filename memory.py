from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from config import MEMORY_DIR
from db import connect
from rag import embed_texts, get_embedding_model


class LongTermMemory:
    """Structured SQLite memory + semantic FAISS memory per student."""

    def __init__(self, student_id: str):
        self.student_id = student_id
        self.root = MEMORY_DIR / student_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "memory.faiss"
        self.meta_path = self.root / "metadata.json"
        self._load()

    def _load(self) -> None:
        if self.index_path.exists() and self.meta_path.exists():
            try:
                index = faiss.read_index(str(self.index_path))
                metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))
                if isinstance(metadata, list) and index.ntotal == len(metadata):
                    self.index = index
                    self.metadata = metadata
                    return
            except Exception:
                pass
        self.index = None
        self.metadata: list[dict[str, Any]] = []

    def _save(self) -> None:
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
        self.meta_path.write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def add(self, content: str, memory_type: str = "learning", subject: str = "", topic: str = "", importance: float = 0.7, confidence: float = 0.8) -> int | None:
        content = content.strip()
        if len(content) < 12:
            return None
        # Avoid exact duplicates.
        if any(m.get("content", "").strip().lower() == content.lower() for m in self.metadata):
            return None
        now = datetime.utcnow().isoformat()
        with connect() as con:
            cur = con.execute("INSERT INTO memories(student_id,memory_type,content,subject,topic,importance,confidence,created_at,updated_at,last_used_at) VALUES(?,?,?,?,?,?,?,?,?,?)", (self.student_id, memory_type, content, subject, topic, importance, confidence, now, now, now))
            memory_id = int(cur.lastrowid)
        vector = embed_texts([content])
        if self.index is None:
            self.index = faiss.IndexFlatIP(vector.shape[1])
        self.index.add(vector)
        self.metadata.append({"memory_id": memory_id, "content": content, "memory_type": memory_type, "subject": subject, "topic": topic, "importance": importance, "confidence": confidence, "created_at": now})
        self._save()
        return memory_id

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self.index is None or self.index.ntotal == 0:
            return []
        q = embed_texts([query])
        scores, ids = self.index.search(q, min(top_k, self.index.ntotal))
        result = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = dict(self.metadata[int(idx)])
            item["score"] = float(score)
            result.append(item)
        return result

    def recent(self, limit: int = 10) -> list[dict[str, Any]]:
        with connect() as con:
            rows = con.execute("SELECT * FROM memories WHERE student_id=? AND is_active=1 ORDER BY id DESC LIMIT ?", (self.student_id, limit)).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict[str, Any]:
        """Return count and approximate on-disk size for this student's semantic memory."""
        vector_bytes = sum(p.stat().st_size for p in self.root.glob("*") if p.is_file())
        with connect() as con:
            row = con.execute(
                "SELECT COUNT(*) AS n FROM memories WHERE student_id=? AND is_active=1",
                (self.student_id,),
            ).fetchone()
        return {
            "count": int(row["n"] if row else 0),
            "vector_bytes": int(vector_bytes),
        }

    def clear(self) -> None:
        with connect() as con:
            con.execute("DELETE FROM memories WHERE student_id=?", (self.student_id,))
        self.index = None
        self.metadata = []
        # Remove the old vector index too. Otherwise a later app run could
        # reload stale vectors with an empty metadata list.
        if self.index_path.exists():
            self.index_path.unlink()
        self._save()


def memory_prompt(memories: list[dict[str, Any]]) -> str:
    if not memories:
        return "No long-term student memories were found."
    lines = ["Relevant long-term student memories:"]
    for m in memories:
        lines.append(f"- [{m.get('memory_type','learning')}] {m.get('content','')}")
    return "\n".join(lines)
