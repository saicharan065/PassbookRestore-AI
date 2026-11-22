import io

import cv2
import numpy as np
import pytest

flask = pytest.importorskip("flask")
from passbookrestore.webapp import create_app  # noqa: E402


@pytest.fixture()
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_index_and_samples(client):
    assert client.get("/").status_code == 200
    names = client.get("/api/samples").get_json()
    assert len(names) > 0


def test_restore_sample_returns_metrics_and_downloads(client):
    name = client.get("/api/samples").get_json()[0]["name"]
    r = client.post("/api/restore", data={"sample": name, "upscaler": "opencv", "scale": "2"})
    assert r.status_code == 200
    j = r.get_json()
    assert j["before"].startswith("data:image") and j["after"].startswith("data:image")
    assert "restored_cer" in j["metrics"]
    for kind in ("restored", "comparison", "text"):
        assert client.get(f"/api/download/{j['id']}/{kind}").status_code == 200


def test_upload_and_bad_inputs(client):
    img = np.full((300, 500, 3), 235, np.uint8)
    cv2.putText(img, "PASSBOOK 12345", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 2)
    ok, buf = cv2.imencode(".png", img)
    r = client.post("/api/restore", data={"file": (io.BytesIO(buf.tobytes()), "scan.png"), "upscaler": "opencv"},
                    content_type="multipart/form-data")
    assert r.status_code == 200 and r.get_json()["has_truth"] is False
    bad = client.post("/api/restore", data={"file": (io.BytesIO(b"nope"), "x.txt")}, content_type="multipart/form-data")
    assert bad.status_code == 400
    assert client.post("/api/restore", data={}).status_code == 400
    assert client.get("/api/download/missing/restored").status_code == 404


def test_history_saved_in_mongo(tmp_path):
    mongomock = pytest.importorskip("mongomock")
    from passbookrestore.webapp.store import RunStore
    app = create_app(RunStore(collection=mongomock.MongoClient().db.runs))
    c = app.test_client()
    assert c.get("/api/history").get_json() == {"enabled": True, "error": None, "runs": []}
    name = c.get("/api/samples").get_json()[0]["name"]
    r = c.post("/api/restore", data={"sample": name, "upscaler": "opencv"}).get_json()
    assert r["run_id"]
    runs = c.get("/api/history").get_json()["runs"]
    assert len(runs) == 1 and runs[0]["filename"] == name and "before" not in runs[0]
    full = c.get(f"/api/history/{r['run_id']}").get_json()
    assert full["before"].startswith("data:image") and "restored_cer" in full["metrics"]
    assert c.delete(f"/api/history/{r['run_id']}").get_json() == {"deleted": True}
    assert c.get("/api/history").get_json()["runs"] == []


def test_app_works_without_mongo():
    from passbookrestore.webapp.store import RunStore
    s = RunStore(uri="mongodb://127.0.0.1:1")      # nothing listens here
    assert not s.enabled and s.error
    c = create_app(s).test_client()
    name = c.get("/api/samples").get_json()[0]["name"]
    r = c.post("/api/restore", data={"sample": name, "upscaler": "opencv"})
    assert r.status_code == 200 and r.get_json()["run_id"] is None
