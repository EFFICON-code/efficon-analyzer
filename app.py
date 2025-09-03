# main.py
import os, io, re
from typing import List, Tuple
from pathlib import Path

import numpy as np
import requests
from flask import Flask, request, Response
from werkzeug.utils import secure_filename

# Lectura de documentos
from pdfminer.high_level import extract_text as pdf_extract_text   # pdfminer.six
from docx import Document as DocxDocument                          # python-docx


# ================== Config ==================
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_API_URL", os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
EMBED_MODEL     = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL      = os.environ.get("CHAT_MODEL",  "gpt-4o-mini")

ALLOWED_EXT = {".pdf", ".docx", ".txt"}  # agrega más si lo necesitas


# ================== Utilidades ==================
def text_response(s: str, status: int = 200) -> Response:
    """Responde siempre texto plano UTF-8."""
    return Response((s or "").strip() + "\n", status=status, mimetype="text/plain; charset=utf-8")

def allowed_file(filename: str) -> bool:
    return Path(filename.lower()).suffix in ALLOWED_EXT

def read_txt_bytes(b: bytes) -> str:
    try:
        return b.decode("utf-8", errors="ignore")
    except Exception:
        return b.decode("latin-1", errors="ignore")

def read_docx_bytes(b: bytes) -> str:
    bio = io.BytesIO(b)
    doc = DocxDocument(bio)
    parts = []
    # Párrafos
    parts.extend([p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()])
    # (opcional) Tablas como texto simple
    for t in doc.tables:
        for row in t.rows:
            parts.append(" | ".join([c.text.strip() for c in row.cells]))
    return "\n".join([p for p in parts if p])

def read_pdf_bytes(b: bytes) -> str:
    bio = io.BytesIO(b)
    return pdf_extract_text(bio) or ""

def extract_text_any(filename: str, content: bytes) -> str:
    ext = Path(filename.lower()).suffix
    if ext == ".pdf":
        return read_pdf_bytes(content)
    if ext == ".docx":
        return read_docx_bytes(content)
    if ext == ".txt":
        return read_txt_bytes(content)
    return ""

def normalize_spaces(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def chunk_text(text: str, max_chars: int = 2500, overlap: int = 250) -> List[str]:
    """Fragmenta por párrafos y une hasta ~max_chars; añade solape para contexto."""
    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    chunks, buf, buf_len = [], [], 0
    for p in paras:
        if buf_len + len(p) + 1 > max_chars and buf:
            chunks.append("\n\n".join(buf))
            tail = chunks[-1][-overlap:] if overlap > 0 else ""
            buf = [tail, p] if tail else [p]
            buf_len = len("".join(buf))
        else:
            buf.append(p)
            buf_len += len(p) + 1
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks

def openai_embed(texts: List[str]) -> np.ndarray:
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": EMBED_MODEL, "input": texts}
    r = requests.post(f"{OPENAI_BASE_URL}/embeddings", headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    vecs = [item["embedding"] for item in data["data"]]
    return np.array(vecs, dtype=np.float32)

def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return np.dot(a_norm, b_norm.T)

def build_prompt_text(instruction: str, selected_chunks: List[Tuple[int, str]]) -> List[dict]:
    """Prompt para respuesta en TEXTO PLANO (sin JSON/Markdown)."""
    corpus = "\n\n".join([f"[Fragmento {i+1}]\n{c}" for i, (_, c) in enumerate(selected_chunks)])
    system = (
        "Eres un analista experto en contratación pública del Ecuador y auditor técnico."
        "\nLee los fragmentos y cumple la instrucción con rigor y precisión."
        "\nResponde en TEXTO PLANO, sin listas, sin títulos, sin Markdown."
        "\nSi se mencionan proformas, integra comparación y conclusión de valor por dinero dentro del mismo texto."
        "\nCita entre corchetes [Fragmento #] solo cuando aporte claridad."
    )
    user = (
        f"Instrucción:\n{instruction}\n\n"
        f"Fragmentos relevantes:\n{corpus}\n\n"
        "Responde en TEXTO PLANO. No uses JSON ni listas."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

def openai_chat(messages: List[dict]) -> str:
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": CHAT_MODEL, "messages": messages, "temperature": 0.2}
    r = requests.post(f"{OPENAI_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


# ================== Flask App ==================
app = Flask(__name__)
# Límite de subida (100 MB). Ajusta si lo necesitas.
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

@app.route("/", methods=["GET"])
def health():
    return text_response("✅ EFFICON Analyzer activo.")

@app.route("/api/analyze", methods=["POST"])
def analyze():
    if not OPENAI_API_KEY:
        return text_response("OPENAI_API_KEY no configurada", 500)

    # --- INSTRUCCIÓN: acepta form-data, cabecera y querystring ---
    instruction = (
        (request.form.get("instruction") or "").strip()
        or (request.headers.get("X-Instruction") or "").strip()
        or (request.args.get("instruction") or "").strip()
    )
    if not instruction:
        dbg = {
            "content_type": request.content_type,
            "form_keys": list(request.form.keys()),
            "file_keys": list(request.files.keys()),
        }
        return text_response(f"Falta 'instruction' (no llegó en form, header ni query). Debug: {dbg}", 400)

    # --- ARCHIVO: preferimos 'file'; aceptamos 'files' por compatibilidad ---
    upfile = request.files.get("file")
    if not upfile and "files" in request.files:
        try:
            upfile = request.files.getlist("files")[0]
        except Exception:
            upfile = None

    if not upfile or not upfile.filename:
        return text_response("Sube un archivo en el campo 'file'", 400)

    filename = secure_filename(upfile.filename)
    if not allowed_file(filename):
        return text_response(f"Extensión no permitida: {Path(filename).suffix}. Usa .pdf, .docx o .txt", 400)

    data = upfile.read()
    if not data:
        return text_response("Archivo vacío.", 400)

    # --- Extracción de texto ---
    try:
        text = extract_text_any(filename, data)
        text = normalize_spaces(text)
    except Exception as e:
        return text_response(f"Error extrayendo texto: {e}", 500)

    if len(text) < 50:
        return text_response("No se pudo extraer texto útil (¿PDF escaneado sin OCR?)", 422)

    # --- Fragmentación y selección por similitud con la instrucción ---
    full_text = f"<<{filename}>>\n{text}"
    chunks = chunk_text(full_text, max_chars=2500, overlap=250)
    try:
        chunk_vecs = openai_embed(chunks)
        instr_vec  = openai_embed([instruction])
        sims = cosine_sim_matrix(instr_vec, chunk_vecs).flatten()
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else 502
        detail = e.response.text if e.response is not None else str(e)
        return text_response(f"Error en embeddings OpenAI ({status}): {detail}", 502)
    except Exception as e:
        return text_response(f"Error inesperado en embeddings: {e}", 502)

    K = min(10, len(chunks))
    top_idx = np.argsort(-sims)[:K]
    selected = [(int(i), chunks[int(i)]) for i in top_idx]

    # --- Chat (respuesta en TEXTO PLANO) ---
    messages = build_prompt_text(instruction, selected)
    try:
        answer = openai_chat(messages)
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else 502
        detail = e.response.text if e.response is not None else str(e)
        return text_response(f"Error en chat OpenAI ({status}): {detail}", 502)
    except Exception as e:
        return text_response(f"Error inesperado en chat: {e}", 502)

    return text_response(answer or "", 200)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    # host 0.0.0.0 para Railway / contenedores
    app.run(host="0.0.0.0", port=port)
