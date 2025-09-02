import os, io, re, json
from typing import List, Tuple
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Lectura de documentos
from pdfminer.high_level import extract_text as pdf_extract_text  # pdfminer.six
from docx import Document as DocxDocument  # python-docx

import numpy as np
import requests

# ===== Config =====
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
EMBED_MODEL = os.environ.get("text-embedding-3-large", "text-embedding-3-small")
CHAT_MODEL  = os.environ.get("gpt-4o",  "gpt-4o-mini")

ALLOWED_EXT = {".pdf", ".docx", ".txt"}

# ===== Utilidades =====
def allowed_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXT

def read_txt_bytes(b: bytes) -> str:
    try:
        return b.decode("utf-8", errors="ignore")
    except Exception:
        return b.decode("latin-1", errors="ignore")

def read_docx_bytes(b: bytes) -> str:
    bio = io.BytesIO(b)
    doc = DocxDocument(bio)
    parts = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(parts)

def read_pdf_bytes(b: bytes) -> str:
    bio = io.BytesIO(b)
    return pdf_extract_text(bio) or ""

def extract_text_any(filename: str, content: bytes) -> str:
    ext = os.path.splitext(filename.lower())[1]
    if ext == ".pdf":
        return read_pdf_bytes(content)
    elif ext == ".docx":
        return read_docx_bytes(content)
    elif ext == ".txt":
        return read_txt_bytes(content)
    return ""

def normalize_spaces(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def chunk_text(text: str, max_chars: int = 2500, overlap: int = 200) -> List[str]:
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

def build_prompt(instruction: str, selected_chunks: List[Tuple[int, str]]) -> List[dict]:
    corpus = "\n\n".join([f"[Fragmento {i+1}]\n{c}" for i, (_, c) in enumerate(selected_chunks)])
    system = (
        "Eres un analista experto en contratación pública del Ecuador y un auditor técnico.\n"
        "Lee los fragmentos y responde a la instrucción con rigor, citando [Fragmento #].\n"
        "Si se menciona proformas, arma una tabla comparativa y conclusión de valor por dinero.\n"
        'Devuelve JSON válido con posibles campos: {"antecedentes":"", "analisis":"", "revision_proformas":{"criterios":[],"tabla":[],"conclusion":""}}'
    )
    user = f"Instrucción:\n{instruction}\n\nFragmentos:\n{corpus}\n\nResponde SOLO en JSON."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def openai_chat(messages: List[dict]) -> str:
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": CHAT_MODEL, "messages": messages, "temperature": 0.2}
    r = requests.post(f"{OPENAI_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

# ===== App =====
app = Flask(__name__)
# Límite de subida (ajústalo según tu caso; 100MB cubre 50 págs escaneadas)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

@app.route("/", methods=["GET"])
def health():
    return "✅ EFFICON Analyzer activo."

@app.route("/api/analyze", methods=["POST"])
def analyze():
    if not OPENAI_API_KEY:
        return jsonify({"error": "OPENAI_API_KEY no configurada"}), 500

    instruction = request.form.get("instruction", "").strip()
    if not instruction:
        return jsonify({"error": "Falta 'instruction'"}), 400

    if "files" not in request.files:
        return jsonify({"error": "Sube al menos un archivo como 'files'"}), 400

    files = request.files.getlist("files")
    docs_texts, meta = [], []

    for f in files:
        filename = secure_filename(f.filename or "archivo")
        data = f.read()
        if not allowed_file(filename):
            return jsonify({"error": f"Extensión no permitida: {filename}"}), 400
        text = extract_text_any(filename, data)
        text = normalize_spaces(text)
        docs_texts.append((filename, text))
        meta.append({"filename": filename, "chars": len(text)})

    merged = []
    for name, t in docs_texts:
        if t:
            merged.append(f"<<{name}>>\n{t}")
    full_text = "\n\n".join(merged)

    if len(full_text) < 50:
        return jsonify({"error": "No se pudo extraer texto útil (¿PDF escaneado sin OCR?)"}), 422

    chunks = chunk_text(full_text, max_chars=2500, overlap=250)
    chunk_vecs = openai_embed(chunks)
    instr_vec  = openai_embed([instruction])
    sims = cosine_sim_matrix(instr_vec, chunk_vecs).flatten()

    K = min(10, len(chunks))
    top_idx = np.argsort(-sims)[:K]
    selected = [(int(i), chunks[int(i)]) for i in top_idx]

    messages = build_prompt(instruction, selected)
    answer = openai_chat(messages)

    try:
        parsed = json.loads(answer)
    except Exception:
        parsed = {"raw": answer}

    out = {
        "status": "ok",
        "meta": meta,
        "used_chunks": sorted([int(i) for i in top_idx.tolist()]),
        "result": parsed
    }
    return jsonify(out), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
