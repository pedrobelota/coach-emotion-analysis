"""Small CPU demo API. Uploads and results are scoped to one request."""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from coach_emotion.demo import MAX_BYTES, MAX_SECONDS, SYNTHETIC_ROWS, analyze_upload


def create_app():
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    from starlette.concurrency import run_in_threadpool

    app = FastAPI(title="Coach Emotion Analysis", docs_url=None, redoc_url=None)
    app.state.pending = 0
    gate = asyncio.Semaphore(1)
    assets = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=assets), name="static")

    @app.middleware("http")
    async def headers(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; img-src 'self' data:; base-uri 'none'; "
            "form-action 'none'; frame-ancestors 'self' https://huggingface.co"
        )
        return response

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(assets / "index.html")

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "model": "whisper-tiny + acoustic SER",
            "device": "cpu",
            "max_seconds": MAX_SECONDS,
            "max_bytes": MAX_BYTES,
        }

    @app.get("/api/example")
    def example():
        return {
            "rows": SYNTHETIC_ROWS,
            "synthetic": True,
            "message": "Invented schema example. No model inference was performed.",
        }

    @app.post("/api/analyze")
    async def analyze(request: Request, consent: bool = False):
        if consent is not True:
            raise HTTPException(400, "Authorization to process the recording is required.")
        if request.headers.get("content-type", "").split(";")[0].lower() not in (
            "audio/wav",
            "audio/x-wav",
            "application/octet-stream",
        ):
            raise HTTPException(415, "Send the raw PCM WAV body with Content-Type: audio/wav.")
        length = request.headers.get("content-length")
        if length is not None:
            try:
                size = int(length)
            except ValueError as exc:
                raise HTTPException(400, "Invalid content length.") from exc
            if size < 0 or size > MAX_BYTES:
                raise HTTPException(413, "Upload exceeds the 8 MiB limit.")
        if app.state.pending >= 4:
            raise HTTPException(429, "The demo queue is full. Try again shortly.")
        app.state.pending += 1
        try:
            async with gate:
                with TemporaryDirectory(prefix="coach_upload_") as folder:
                    path = Path(folder) / "upload.wav"
                    received = 0
                    try:
                        async with asyncio.timeout(30):
                            with path.open("wb") as output:
                                async for chunk in request.stream():
                                    received += len(chunk)
                                    if received > MAX_BYTES:
                                        raise HTTPException(413, "Upload exceeds the 8 MiB limit.")
                                    output.write(chunk)
                    except TimeoutError:
                        raise HTTPException(408, "Upload timed out.") from None
                    try:
                        rows, message = await run_in_threadpool(analyze_upload, str(path), True)
                    except ValueError as exc:
                        raise HTTPException(422, str(exc)) from None
                    except (RuntimeError, OSError):
                        raise HTTPException(
                            503,
                            "Inference could not complete. Try a shorter valid WAV or retry after model download.",
                        ) from None
                    return {"rows": rows, "message": message, "synthetic": False}
        finally:
            app.state.pending -= 1

    return app


def main():
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
