# app_audio/main.py
# ---------------------------------------------------------------------------
# Micro-STT para encuestas canal 5 (audio grabado)
#   • POST /stt          → transcribe, extrae respuestas y las envía al backend
#   • POST /whisper-test → solo transcripción Whisper (debug)
# ---------------------------------------------------------------------------
import os, io, uuid, json, logging, tempfile, requests
from typing import Dict, Any, List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from openai import OpenAI

# ───────────────────────── Config ────────────────────────────────────────
load_dotenv()

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL")         # ej. http://localhost:8000

if not OPENAI_API_KEY or not BACKEND_BASE_URL:
    raise RuntimeError("Necesitas OPENAI_API_KEY y BACKEND_BASE_URL en el .env")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
app = FastAPI(title="Micro-STT encuestas audio (canal 5)")

# ───────────────────────── Helpers ───────────────────────────────────────
def fetch_plantilla(entrega_id: str) -> Dict[str, Any]:
    url = f"https://{BACKEND_BASE_URL}/public/entregas/{entrega_id}/plantilla-mapa"
    r   = requests.get(url, timeout=10)
    if r.status_code != 200:
        raise HTTPException(r.status_code, f"Backend no devolvió plantilla ({r.text})")
    return r.json()

def _safe_suffix(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext in {".flac",".m4a",".mp3",".mp4",".mpeg",".mpga",".oga",".ogg",".wav",".webm"}:
        return ext
    return ".wav"

def transcribe_openai(raw_audio: bytes, original_name: str) -> str:
    suffix = _safe_suffix(original_name)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw_audio)
        tmp.flush()
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as f:
            resp = openai_client.audio.transcriptions.create(
                file=f,
                model="whisper-1",
                response_format="text",
                language="es",
            )
        return resp.strip()
    finally:
        os.remove(tmp_path)

# ───────────── Sanitizado y utilidades ───────────────────────────────────
def sanitize_answers(data: Dict[str, Any], plantilla: Dict[str, Any]) -> Dict[str, Any]:
    mapa = {p["id"]: p for p in plantilla["preguntas"]}
    dep: List[Dict[str, Any]] = []
    for item in data.get("respuestas_preguntas", []):
        # Aceptamos 'pregunta_id' o 'id'
        if not item.get("pregunta_id") and item.get("id"):
            item["pregunta_id"] = item.pop("id")

        qid  = item.get("pregunta_id")
        if not qid:
            continue

        tipo = item.get("tipo_pregunta_id")
        preg = mapa.get(qid) or {}

        # tipo 1 – quitar duplicados
        if tipo == 1 and item.get("texto"):
            if item["texto"].strip().lower() == preg.get("texto", "").lower():
                item["texto"] = None

        # tipo 2 – solo número
        if tipo == 2 and item.get("numero") is not None:
            item["texto"] = None

        if tipo == 3:
            item["texto"] = None
            if item.get("opciones_ids"):
                item["opcion_id"] = item["opciones_ids"][0]
                item["opciones_ids"] = []

        if tipo == 4:
            item["texto"] = None
            valid = {o["id"] for o in preg.get("opciones", [])}
            item["opciones_ids"] = [oid for oid in item.get("opciones_ids", []) if oid in valid]

        dep.append(item)

    data["respuestas_preguntas"] = dep
    return data

def build_backend_payload(res_json: Dict[str, Any]) -> Dict[str, Any]:
    filas: List[Dict[str, Any]] = []
    for r in res_json.get("respuestas_preguntas", []):
        base: Dict[str, Any] = {"pregunta_id": r["pregunta_id"]}

        if txt := r.get("texto"):
            base["texto"] = txt
        if r.get("numero") not in (None, ""):
            base["numero"] = r["numero"]
        if oid := r.get("opcion_id"):
            base["opcion_id"] = oid

        if r.get("opciones_ids"):
            for oid in r["opciones_ids"]:
                filas.append({**base, "opcion_id": oid})
        else:
            filas.append(base)

    return {"respuestas_preguntas": filas}

@app.post("/stt")
async def stt_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("audio/"):
        raise HTTPException(400, "Sube un archivo de audio (wav/mp3/ogg/webm…)")

    audio_bytes = await file.read()

    try:
        entrega_id = file.filename.split(".")[0]
        uuid.UUID(entrega_id)                       
    except Exception:
        raise HTTPException(400, "El archivo debe llamarse <entrega_id>.wav/.mp3/etc")

    plantilla = fetch_plantilla(entrega_id)
    preguntas = plantilla["preguntas"]

    transcript = transcribe_openai(audio_bytes, file.filename)
    logging.info("📝 Whisper → %s caracteres", len(transcript))

    prompt_gpt = (
        "Eres un extractor de respuestas para encuestas capturadas en audio.No te inventes respuestas , tira error si no hay nada en el audio\n"
        "Devuelve SOLO JSON con la clave «respuestas_preguntas» (lista).\n"
        "Cada elemento DEBE tener estas claves:\n"
        "  pregunta_id, tipo_pregunta_id, texto, numero, opcion_id, opciones_ids\n"
        "Reglas:\n"
        " • tipo 1 → poner la respuesta en «texto».\n"
        " • tipo 2 → poner un número en «numero».\n"
        " • tipo 3 → EXACTAMENTE un UUID en «opcion_id» (lista vacía).\n"
        " • tipo 4 → lista «opciones_ids» con los UUID marcados.\n\n"
        f"Plantilla de preguntas:\n{json.dumps(preguntas, ensure_ascii=False)}\n\n"
        "Transcripción íntegra del audio del encuestado:\n"
        f"{transcript}"
    )

    gpt = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Asistente para extraer respuestas de la transcripción."},
            {"role": "user",   "content": prompt_gpt},
        ],
        response_format={"type": "json_object"},
        max_tokens=1500,
    )

    try:
        raw_json = json.loads(gpt.choices[0].message.content.strip())
    except Exception as e:
        raise HTTPException(500, f"GPT no devolvió JSON válido: {e}")

    ocr_json        = sanitize_answers(raw_json, plantilla)
    backend_payload = build_backend_payload(ocr_json)

    try:
        url  = f"https://{BACKEND_BASE_URL}/public/entregas/{entrega_id}/respuestas"
        resp = requests.post(url, json=backend_payload, timeout=15)
        resp.raise_for_status()
        backend_resp = resp.json()
    except Exception as exc:
        backend_resp = {"error": str(exc), "payload_enviado": backend_payload}

    return JSONResponse({
        "entrega_id":        entrega_id,
        "transcripcion":     transcript,
        "payload_enviado":   backend_payload,
        "respuesta_backend": backend_resp,
    })

# ─────────────────────── Endpoint /whisper-test ──────────────────────────
@app.post("/whisper-test")
async def whisper_test(file: UploadFile = File(...)):
    if not file.content_type.startswith("audio/"):
        raise HTTPException(400, "Debes subir un archivo de audio")
    audio = await file.read()

    try:
        texto = transcribe_openai(audio, file.filename)
    except Exception as exc:
        raise HTTPException(500, f"Error transcribiendo: {exc}")

    return {"filename": file.filename, "transcripcion": texto}

# ────────────────────────── Health-check ─────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}
