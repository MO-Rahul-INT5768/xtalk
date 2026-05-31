import argparse
import json
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, Form, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import mimetypes

mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")
import logging
from xtalk import Xtalk
from xtalk.log_utils import mute_other_logging

mute_other_logging()
# Add dedicated DEBUG file handler directly to xtalk logger
_xtalk_logger = logging.getLogger("xtalk")
_xtalk_logger.setLevel(logging.DEBUG)
_debug_handler = logging.FileHandler("/home/sagemaker-user/xtalk/logs/xtalk_pipeline_debug.log")
_debug_handler.setLevel(logging.DEBUG)
_debug_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
_xtalk_logger.addHandler(_debug_handler)

parser = argparse.ArgumentParser(description="Configurable Xtalk Server")
parser.add_argument("--config", type=str, help="Path to the server configuration file")
parser.add_argument("--port", type=int, help="Port number for the server to listen on")
args = parser.parse_args()

app = FastAPI(title="Xtalk Server")

# Instantiate Xtalk from config
# config can be passed as a path to json file or a dict
xtalk_instance = Xtalk.from_config(args.config)


# Mount WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await xtalk_instance.connect(websocket)


# Serve static files
example_server_path = Path(__file__).parent
templates = Jinja2Templates(directory=str(example_server_path / "templates"))
static_root = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_root)), name="static")
try: 
    app.mount(
        "/xtalk",
        StaticFiles(
            directory=str(Path(__file__).parent.parent.parent / "frontend" / "dist")
        ),
        name="xtalk",
    )
except:
    print("No local X-Talk frontend library found. You may use the library from CDN.")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/modern", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index_modern.html", {"request": request})


# Mount text embedding endpoint
@app.post("/api/upload")
async def upload_file(
    session_id: str = Form(...),
    file: UploadFile = File(...),
):
    # Check file type
    content_type = (file.content_type or "").lower()
    filename = (file.filename or "").lower()
    is_text = content_type.startswith("text/") if content_type else False
    if content_type and not is_text:
        raise HTTPException(status_code=400, detail="Only text files are supported.")
    # Read file content and embed
    text = (await file.read()).decode("utf-8", errors="ignore")
    await xtalk_instance.embed_text(session_id=session_id, text=text)
    return {"status": "ok"}


# Mount voices
@app.get("/api/voices")
async def get_reference_audios():
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
        try:
            voices = config["tts"]["params"]["voices"]
        except:
            voices = []
    return JSONResponse(content={"audios": voices})


@app.get("/api/available-tts-models")
async def get_available_tts_models():
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    tts_cfg = config.get("tts", {})
    model_type = tts_cfg.get("type", "")
    model_params = tts_cfg.get("params", {})
    display_name = model_params.get("model", model_type) or model_type
    return JSONResponse(content={"models": [{"type": model_type, "name": display_name, "config": model_params}]})


@app.get("/api/available-llm-models")
async def get_available_llm_models():
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    llm_cfg = config.get("llm_agent", {}).get("params", {}).get("model", {})
    model_name = llm_cfg.get("model", "")
    base_url = llm_cfg.get("base_url", "")
    api_key = llm_cfg.get("api_key", "")
    return JSONResponse(content={"models": [{"model": model_name, "display_name": model_name, "base_url": base_url, "api_key": api_key}]})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=args.port or 11995)
