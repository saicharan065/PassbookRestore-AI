"""Optional MongoDB persistence for web-app restoration runs.

Set MONGODB_URI (default mongodb://localhost:27017) and MONGODB_DB (default passbookrestore).
If MongoDB is unreachable the web app keeps working and simply shows no history.
"""

from __future__ import annotations

import datetime as dt
import logging
import os

log = logging.getLogger("passbookrestore")


class RunStore:
    def __init__(self, collection=None, uri: str | None = None, db_name: str | None = None):
        self.collection = collection
        self.error: str | None = None
        if collection is None:
            self._connect(uri or os.environ.get("MONGODB_URI", "mongodb://localhost:27017"),
                          db_name or os.environ.get("MONGODB_DB", "passbookrestore"))

    def _connect(self, uri: str, db_name: str) -> None:
        try:
            from pymongo import MongoClient
            client = MongoClient(uri, serverSelectionTimeoutMS=2000)
            client.admin.command("ping")
            self.collection = client[db_name]["runs"]
            self.collection.create_index("created_at")
            log.info("MongoDB connected: %s / %s", uri, db_name)
        except Exception as exc:
            self.collection = None
            self.error = f"{type(exc).__name__}: {exc}"
            log.warning("MongoDB unavailable (%s); run history disabled.", self.error)

    @property
    def enabled(self) -> bool:
        return self.collection is not None

    def save(self, doc: dict) -> str | None:
        if not self.enabled:
            return None
        try:
            doc = {**doc, "created_at": dt.datetime.now(dt.timezone.utc)}
            return str(self.collection.insert_one(doc).inserted_id)
        except Exception as exc:
            log.warning("could not save run: %s", exc)
            return None

    def list(self, limit: int = 50) -> list[dict]:
        """Newest first, without the heavy fields (images and text)."""
        if not self.enabled:
            return []
        proj = {"before": 0, "after": 0, "baseline_text": 0, "restored_text": 0, "truth": 0}
        out = []
        for d in self.collection.find({}, proj).sort("created_at", -1).limit(limit):
            d["_id"] = str(d["_id"])
            d["created_at"] = d["created_at"].isoformat()
            out.append(d)
        return out

    def get(self, run_id: str) -> dict | None:
        if not self.enabled:
            return None
        from bson import ObjectId
        try:
            d = self.collection.find_one({"_id": ObjectId(run_id)})
        except Exception:
            return None
        if d:
            d["_id"] = str(d["_id"])
            d["created_at"] = d["created_at"].isoformat()
        return d

    def delete(self, run_id: str) -> bool:
        if not self.enabled:
            return False
        from bson import ObjectId
        try:
            return self.collection.delete_one({"_id": ObjectId(run_id)}).deleted_count == 1
        except Exception:
            return False
