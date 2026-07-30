from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .db import CandidateStore
from .image_utils import overlay_png


APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT_BASE = REPO_ROOT / "outputs/stage2/overlap_audit_20260622"
DEFAULT_MANIFEST_ROOT = REPO_ROOT / "manifests/stage2_v2_strict_v1"
DEFAULT_REVIEWED_ROOT = REPO_ROOT / "manifests/stage2_v2_strict_v1_overlap_reviewed"


class Decision(BaseModel):
    review_label: str
    review_action: str
    comment: str = Field(default="", max_length=2000)


def create_app(audit_root: Path | None = None, audit_base: Path | None = None) -> FastAPI:
    if audit_root is not None:
        stores = {audit_root.name: CandidateStore(audit_root)}
    elif os.environ.get("OVERLAP_AUDIT_ROOT"):
        root = Path(os.environ["OVERLAP_AUDIT_ROOT"])
        stores = {root.name: CandidateStore(root)}
    else:
        base = audit_base or Path(os.environ.get("OVERLAP_AUDIT_BASE", DEFAULT_AUDIT_BASE))
        stores = {
            protocol: CandidateStore(base / protocol)
            for protocol in ("legacy", "strict_v1")
            if (base / protocol / "candidates_all.csv").is_file()
        }
        if not stores:
            raise FileNotFoundError(f"no audit protocols under {base}")
    default_protocol = "strict_v1" if "strict_v1" in stores else next(iter(stores))
    apply_lock = threading.Lock()
    app = FastAPI(title="OverlapReview-QC", version="1.1")
    app.state.stores = stores
    app.mount("/static", StaticFiles(directory=APP_ROOT / "static"), name="static")

    def select_store(protocol: str | None) -> tuple[str, CandidateStore]:
        selected = protocol or default_protocol
        if selected not in stores:
            raise HTTPException(status_code=404, detail=f"unknown protocol: {selected}")
        return selected, stores[selected]

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(APP_ROOT / "templates/index.html")

    @app.get("/api/summary")
    def summary(protocol: str | None = None) -> dict[str, object]:
        selected, store = select_store(protocol)
        return {"protocol": selected, "available_protocols": list(stores), **store.summary()}

    @app.get("/api/candidates")
    def candidates(
        split_pair: str | None = None,
        risk: str | None = None,
        status: str | None = None,
        offset: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=500),
        protocol: str | None = None,
    ) -> dict[str, object]:
        _, store = select_store(protocol)
        return store.query(split_pair, risk, status, offset, limit)

    @app.get("/api/candidates/{pair_id}")
    def candidate(pair_id: str, protocol: str | None = None) -> dict[str, str]:
        _, store = select_store(protocol)
        try:
            return store.merged(pair_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="pair not found") from error

    @app.post("/api/candidates/{pair_id}/decision")
    def decision(pair_id: str, body: Decision, protocol: str | None = None) -> dict[str, str]:
        _, store = select_store(protocol)
        try:
            return store.save_decision(pair_id, body.review_label, body.review_action, body.comment)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="pair not found") from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/api/image/{pair_id}/{side}/{kind}")
    def image(pair_id: str, side: str, kind: str, protocol: str | None = None) -> Response:
        if side not in {"a", "b"} or kind not in {"pre", "sar", "mask", "overlay"}:
            raise HTTPException(status_code=400, detail="invalid side/kind")
        _, store = select_store(protocol)
        try:
            row = store.merged(pair_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="pair not found") from error
        field = {"pre": "pre", "sar": "sar", "mask": "mask"}.get(kind)
        if field:
            path = Path(row[f"{field}_{side}"])
            if not path.is_file():
                raise HTTPException(status_code=404, detail=f"missing image: {path}")
            return FileResponse(path)
        try:
            payload = overlay_png(Path(row[f"pre_{side}"]), Path(row[f"mask_{side}"]))
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return Response(payload, media_type="image/png")

    @app.post("/api/export_decisions")
    def export_decisions(protocol: str | None = None) -> dict[str, object]:
        _, store = select_store(protocol)
        return {"path": str(store.decisions_path), "decision_count": len(store.decisions)}

    @app.post("/api/apply_decisions")
    def apply_decisions(protocol: str | None = None) -> dict[str, object]:
        selected, store = select_store(protocol)
        if selected != "strict_v1":
            raise HTTPException(status_code=409, detail="Only strict_v1 decisions may create a training manifest.")
        out_root = Path(os.environ.get("OVERLAP_REVIEWED_MANIFEST_ROOT", DEFAULT_REVIEWED_ROOT))
        if out_root.exists():
            summary_path = out_root / "overlap_review_summary.json"
            if summary_path.is_file():
                return {
                    "status": "already_exists",
                    "out_root": str(out_root),
                    "summary": summary_path.read_text(encoding="utf-8"),
                }
            raise HTTPException(status_code=409, detail=f"output exists and is not a completed export: {out_root}")
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts/apply_overlap_decisions.py"),
            "--manifest-root", str(DEFAULT_MANIFEST_ROOT),
            "--candidates", str(store.candidates_path),
            "--decisions", str(store.decisions_path),
            "--policy", "relaxed_train_test_only",
            "--out-root", str(out_root),
        ]
        with apply_lock:
            try:
                result = subprocess.run(
                    command, cwd=REPO_ROOT, check=True, capture_output=True, text=True, timeout=120
                )
            except subprocess.CalledProcessError as error:
                detail = (error.stderr or error.stdout or str(error)).strip()
                raise HTTPException(status_code=409, detail=detail) from error
            except subprocess.TimeoutExpired as error:
                raise HTTPException(status_code=504, detail="manifest export timed out") from error
        return {"status": "created", "out_root": str(out_root), "message": result.stdout.strip()}

    return app


try:
    app = create_app()
except FileNotFoundError as startup_error:
    # Keep the module importable before the long audit has produced candidates.
    startup_error_text = str(startup_error)
    app = FastAPI(title="OverlapReview-QC", version="1.0")

    @app.get("/")
    def not_ready() -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": startup_error_text})
