import io
import zipfile
import tempfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

app = FastAPI(title="DXA AI Metrics", version="1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/process")
async def process(file: UploadFile = File(...)):
    work = Path(tempfile.mkdtemp(prefix="dxa_"))
    archive = work / "input.zip"
    archive.write_bytes(await file.read())

    data_dir = work / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive, "r") as z:
        z.extractall(data_dir)

    cfg = Path("/app/config/input.txt")
    cfg.write_text(str(data_dir), encoding="utf-8")

    import runpy
    runpy.run_path("/app/Final_sum.py", run_name="__main__")

    out = Path("/app/report.xlsx")
    if not out.exists():
        return JSONResponse({"error": "report.xlsx не создан"}, status_code=500)

    return FileResponse(
        str(out),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="report.xlsx",
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)