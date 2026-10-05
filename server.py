import os
import sys
import uuid
import shutil
import asyncio
import subprocess
import threading
import json
import re
import logging
import time
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
import static_ffmpeg

static_ffmpeg.add_paths()

logging.getLogger("asyncio").setLevel(logging.CRITICAL)
logging.getLogger("uvicorn.error").setLevel(logging.CRITICAL)
logging.getLogger("uvicorn.access").setLevel(logging.CRITICAL)

if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(
            asyncio.WindowsSelectorEventLoopPolicy()
        )
    except Exception:
        pass

FFMPEG_BIN = "ffmpeg"
try:
    import imageio_ffmpeg
    f_path = imageio_ffmpeg.get_ffmpeg_exe()
    if os.path.exists(f_path):
        FFMPEG_BIN = f_path
except Exception:
    pass

def get_best_video_encoder():
    try:
        res = subprocess.run([FFMPEG_BIN, "-encoders"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        if "h264_nvenc" in res.stdout:
            return "h264_nvenc"
        elif "h264_qsv" in res.stdout:
            return "h264_qsv"
    except Exception:
        pass
    return "libx264"

from fastapi import (
    FastAPI, UploadFile, File, Form, HTTPException, Cookie, Response, Request
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

class BatchRenderConfig(BaseModel):
    session_id: Optional[str] = "default_user"
    episode_ids: List[str]
    merge_all_into_one: bool = False
    speaker_mode: str = "auto"
    speed_factor: float = 1.0
    dub_volume: float = 2.20
    keep_bgm: Optional[bool] = True
    kiri_api_key: Optional[str] = ""
    resolution: Optional[str] = "4k"
    anti_copyright: Optional[bool] = False
    blur_items: Optional[List[dict]] = []
    text_items: Optional[List[dict]] = []
    enable_logo: Optional[bool] = False
    logo_filename: Optional[str] = ""
    logo_norm_x: Optional[float] = 0.15
    logo_norm_y: Optional[float] = 0.10
    logo_norm_w: Optional[float] = 0.40
    dialogues_map: Optional[dict] = None

app = FastAPI(title="KhmerDub Pro Studio", version="77000.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = BASE_DIR / "workspace"
STATIC_DIR = BASE_DIR / "public"
LICENSES_FILE = BASE_DIR / "licenses.json"

WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/workspace", StaticFiles(directory=str(WORKSPACE_ROOT)), name="workspace")
if STATIC_DIR.exists():
    app.mount("/public", StaticFiles(directory=str(STATIC_DIR)), name="public")

RENDER_LOCK = threading.Lock()
TELEGRAM_BOT_TOKEN = "8804336135:AAFjLuRmc3eXsqkVyihL0ZN0-o3wxQ9GvSk"
TELEGRAM_ADMIN_CHAT_ID = "7176722918"

def load_licenses():
    if LICENSES_FILE.exists():
        try:
            with open(LICENSES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"clients": {}, "requests": {}}

def save_licenses(data):
    try:
        with open(LICENSES_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[!] Save License Error: {e}")

def is_service_active(client_sid: str, service_key: str) -> bool:
    db = load_licenses()
    client = db.get("clients", {}).get(client_sid, {})
    sub = client.get(service_key)
    if not sub:
        return False
    if sub.get("plan") == "lifetime":
        return True
    if sub.get("plan") == "month":
        return time.time() < sub.get("expires_at", 0)
    return False

def send_telegram_receipt_photo(req_id: str, client_sid: str, service_name: str, plan_type: str, price: str, receipt_path: Path):
    caption_html = (
        f"🧾 <b>មានវិក្កយបត្របង់ប្រាក់ថ្មី!</b>\n\n"
        f"👤 <b>Client ID:</b> <code>{client_sid}</code>\n"
        f"🛠 <b>សេវាកម្ម:</b> <b>{service_name}</b>\n"
        f"📦 <b>កញ្ចប់:</b> <b>{plan_type}</b>\n"
        f"💵 <b>ចំនួនទឹកប្រាក់:</b> <b>{price}</b>\n"
        f"🆔 <b>Request ID:</b> <code>{req_id}</code>\n"
        f"🕒 <b>ម៉ោង:</b> <code>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>\n\n"
        f"👉 សូមពិនិត្យផ្ទៀងផ្ទាត់វិក្កយបត្រ រួចចុច Approve ឬ Reject ខាងក្រោម!"
    )
    keyboard = {
        "inline_keyboard": [
            [
                {"text": "✅ Approve (យល់ព្រម)", "callback_data": f"approve_{req_id}"},
                {"text": "❌ Reject (បដិសេធ)", "callback_data": f"reject_{req_id}"}
            ]
        ]
    }
    keyboard_json = json.dumps(keyboard)
    url_photo = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    url_msg = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    sent_success = False

    if receipt_path.exists():
        try:
            with open(str(receipt_path), "rb") as img_file:
                files = {"photo": (receipt_path.name, img_file, "image/jpeg")}
                data = {
                    "chat_id": str(TELEGRAM_ADMIN_CHAT_ID),
                    "caption": caption_html,
                    "parse_mode": "HTML",
                    "reply_markup": keyboard_json
                }
                resp = requests.post(url_photo, data=data, files=files, timeout=20)
                if resp.status_code == 200:
                    sent_success = True
        except Exception:
            pass

    if not sent_success:
        try:
            data_text = {
                "chat_id": str(TELEGRAM_ADMIN_CHAT_ID),
                "text": caption_html + f"\n\n⚠️ <i>(រូបភាពរក្សាទុកក្នុង Server: {receipt_path.name})</i>",
                "parse_mode": "HTML",
                "reply_markup": keyboard_json
            }
            resp_msg = requests.post(url_msg, json=data_text, timeout=15)
            if resp_msg.status_code == 200:
                sent_success = True
        except Exception:
            pass
    return sent_success

def telegram_polling_worker():
    offset = 0
    print("[*] Telegram Polling Worker is ACTIVE and listening for approvals...")
    while True:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates?offset={offset}&timeout=10"
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("result", []):
                    offset = item["update_id"] + 1
                    if "callback_query" not in item:
                        continue
                    cb = item["callback_query"]
                    cb_id = cb.get("id")
                    cb_data = cb.get("data", "")
                    message = cb.get("message", {})
                    msg_id = message.get("message_id")

                    try:
                        requests.post(
                            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/answerCallbackQuery",
                            json={"callback_query_id": cb_id, "text": "ដំណើរការជោគជ័យ!"},
                            timeout=5
                        )
                    except Exception:
                        pass

                    db = load_licenses()

                    if cb_data.startswith("approve_"):
                        req_id = cb_data.replace("approve_", "")
                        req_info = db.get("requests", {}).get(req_id, {})
                        
                        sid = req_info.get("client_sid", "default_user")
                        s_key = req_info.get("service_key", "studio")
                        p_type = req_info.get("plan_type", "month")
                        
                        if sid not in db["clients"]:
                            db["clients"][sid] = {}
                            
                        expires = time.time() + (30 * 24 * 3600) if p_type == "month" else 0
                        db["clients"][sid][s_key] = {
                            "plan": p_type,
                            "approved_at": time.time(),
                            "expires_at": expires
                        }
                        
                        if req_id not in db["requests"]:
                            db["requests"][req_id] = {}
                        db["requests"][req_id]["status"] = "approved"
                        db["requests"][req_id]["plan_type"] = p_type
                        
                        save_licenses(db)
                        print(f"[+] APPROVED: {req_id} for client: {sid}")

                        if msg_id:
                            requests.post(
                                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/editMessageCaption",
                                json={
                                    "chat_id": TELEGRAM_ADMIN_CHAT_ID,
                                    "message_id": msg_id,
                                    "caption": f"✅ <b>បានយល់ព្រម (APPROVED) ជោគជ័យ!</b>\n👤 Client: <code>{sid}</code>\n🛠 សេវា: <b>{s_key}</b> ({p_type})\n🆔 Request ID: <code>{req_id}</code>\n🕒 ម៉ោង: <code>{datetime.now().strftime('%H:%M:%S')}</code>",
                                    "parse_mode": "HTML"
                                },
                                timeout=10
                            )

                    elif cb_data.startswith("reject_"):
                        req_id = cb_data.replace("reject_", "")
                        if req_id in db.get("requests", {}):
                            db["requests"][req_id]["status"] = "rejected"
                        save_licenses(db)
                        print(f"[-] REJECTED: {req_id}")

                        if msg_id:
                            requests.post(
                                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/editMessageCaption",
                                json={
                                    "chat_id": TELEGRAM_ADMIN_CHAT_ID,
                                    "message_id": msg_id,
                                    "caption": f"❌ <b>បានបដិសេធ (REJECTED)</b> វិក្កយបត្រនេះ!\n🆔 Request ID: <code>{req_id}</code>\n🕒 ម៉ោង: <code>{datetime.now().strftime('%H:%M:%S')}</code>",
                                    "parse_mode": "HTML"
                                },
                                timeout=10
                            )
        except Exception:
            pass
        time.sleep(1)

threading.Thread(target=telegram_polling_worker, daemon=True).start()

def get_session_workspace(session_id: str):
    sid = re.sub(r"[^a-zA-Z0-9_-]", "", session_id) or "default_user"
    user_dir = WORKSPACE_ROOT / sid
    uploads = user_dir / "uploads"
    processed = user_dir / "processed"
    uploads.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    db_file = user_dir / "episodes.json"
    return user_dir, uploads, processed, db_file

def load_user_db(db_file: Path):
    if db_file.exists():
        try:
            with open(db_file, "r", encoding="utf-8") as f:
                d = json.load(f)
                if isinstance(d, dict):
                    return d
        except Exception:
            pass
    return {}

def save_user_db(db_file: Path, data: dict):
    try:
        with open(db_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[!] Save DB Error: {e}")

def sanitize_url(raw_url: str) -> str:
    u = str(raw_url).strip()
    match = re.search(r"https?://[^\s\[\]\(\)\'\"]+", u)
    return match.group(0).strip() if match else u

def get_media_duration_and_size(file_path: str):
    dur, w, h = 0.0, 1080, 1920
    try:
        cmd = [FFMPEG_BIN, "-i", str(file_path)]
        res = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True, errors="replace")
        d_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", res.stderr)
        if d_match:
            hh, mm, ss = d_match.groups()
            dur = int(hh) * 3600 + int(mm) * 60 + float(ss)
        s_match = re.search(r"Video:.*,\s*(\d{2,5})x(\d{2,5})", res.stderr)
        if s_match:
            w = int(s_match.group(1))
            h = int(s_match.group(2))
    except Exception:
        pass
    return dur, w, h

HTTP_SESSION = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=30, pool_maxsize=50, max_retries=2)
HTTP_SESSION.mount("https://", adapter)
HTTP_SESSION.mount("http://", adapter)

def robust_translate_to_khmer(text: str) -> str:
    clean_t = str(text or "").strip()
    if not clean_t:
        return ""

    try:
        url = "https://translate.googleapis.com/translate_a/single"
        params = {"client": "gtx", "sl": "auto", "tl": "km", "dt": "t", "q": clean_t}
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = HTTP_SESSION.get(url, params=params, headers=headers, timeout=4)
        if resp.status_code == 200:
            data = resp.json()
            if data and data[0]:
                out = "".join([p[0] for p in data[0] if p and p[0]]).strip()
                if re.search(r"[\u1780-\u17ff]", out):
                    return out
    except Exception:
        pass

    try:
        url_dict = "https://clients5.google.com/translate_a/t"
        params_dict = {"client": "dict-chrome-ex", "sl": "auto", "tl": "km", "q": clean_t}
        headers_dict = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp_dict = HTTP_SESSION.get(url_dict, params=params_dict, headers=headers_dict, timeout=4)
        if resp_dict.status_code == 200:
            d_json = resp_dict.json()
            if isinstance(d_json, list) and len(d_json) > 0:
                res_txt = d_json[0] if isinstance(d_json[0], str) else d_json[0][0]
                if re.search(r"[\u1780-\u17ff]", res_txt):
                    return res_txt
    except Exception:
        pass

    try:
        q_enc = urllib.parse.quote(clean_t[:250])
        url_mm = f"https://api.mymemory.translated.net/get?q={q_enc}&langpair=auto|km"
        resp_mm = HTTP_SESSION.get(url_mm, timeout=4)
        if resp_mm.status_code == 200:
            trans = resp_mm.json().get("responseData", {}).get("translatedText", "")
            if re.search(r"[\u1780-\u17ff]", trans) and "MYMEMORY" not in trans:
                return trans
    except Exception:
        pass

    return clean_t

def call_ai_chunk(chunk_items: List[dict], api_key: str, model_type: str, source_lang: str) -> List[dict]:
    if not chunk_items:
        return []
    clean_key = str(api_key or "").strip()

    prompt = (
        "You are an expert film dubbing director.\n"
        "Translate each line into natural spoken Khmer script (អក្សរខ្មែរ).\n"
        "Strict Requirement: Output natural spoken Khmer lines matching the original emotion.\n"
        "Output ONLY a JSON array: [{\"id\": 1, \"speaker_id\": \"S1\", \"role\": \"female\", \"khmer_text\": \"...\"}]"
    )

    if clean_key.startswith("AIza") or model_type == "gemini":
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={clean_key}"
            req_body = {
                "contents": [{"role": "user", "parts": [{"text": f"{prompt}\n\nDialogues to translate:\n{json.dumps(chunk_items, ensure_ascii=False)}"}]}],
                "generationConfig": {"temperature": 0.2}
            }
            resp = HTTP_SESSION.post(url, json=req_body, timeout=20)
            if resp.status_code == 200:
                parts = resp.json()["candidates"][0]["content"]["parts"]
                raw_text = "".join([p.get("text", "") for p in parts]).strip()
                match = re.search(r"\[\s*\{.*\}\s*\]", raw_text, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
        except Exception as e:
            print("[Gemini Translation Error]:", e)

    if model_type == "mwapi" or clean_key.startswith("sk-"):
        endpoints = [
            "https://api.openai.com/v1/chat/completions",
            "https://api.mwapi.net/v1/chat/completions"
        ]
        for ep_url in endpoints:
            try:
                headers = {"Authorization": f"Bearer {clean_key}", "Content-Type": "application/json"}
                payload = {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": f"Translate these to Khmer:\n{json.dumps(chunk_items, ensure_ascii=False)}"}
                    ],
                    "temperature": 0.2
                }
                resp = HTTP_SESSION.post(ep_url, json=payload, headers=headers, timeout=20)
                if resp.status_code == 200:
                    raw = resp.json()["choices"][0]["message"]["content"]
                    match = re.search(r"\[\s*\{.*\}\s*\]", raw, re.DOTALL)
                    if match:
                        return json.loads(match.group(0))
            except Exception:
                pass

    return []

def separate_and_purge_original_vocals(video_in: str, out_bgm_wav: Path) -> bool:
    temp_dir = out_bgm_wav.parent / f"bgm_sep_{uuid.uuid4().hex[:6]}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    extracted_audio = temp_dir / "temp_audio.wav"
    subprocess.run([
        FFMPEG_BIN, "-y", "-i", str(video_in),
        "-vn", "-ar", "44100", "-ac", "2",
        str(extracted_audio)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    dsp_filter = "pan=stereo|c0=c0-c1|c1=c1-c0,highpass=f=120,lowpass=f=12000,volume=1.20"
    cmd_dsp = [
        FFMPEG_BIN, "-y", "-i", str(extracted_audio if extracted_audio.exists() else video_in),
        "-af", dsp_filter,
        "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le",
        str(out_bgm_wav)
    ]
    subprocess.run(cmd_dsp, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    shutil.rmtree(temp_dir, ignore_errors=True)
    return out_bgm_wav.exists() and out_bgm_wav.stat().st_size > 1000

def detect_speaker_acoustic_pitch(audio_path: str, start_sec: float, end_sec: float) -> str:
    try:
        dur = max(0.3, end_sec - start_sec)
        cmd = [
            FFMPEG_BIN, "-y",
            "-ss", f"{start_sec:.3f}",
            "-i", str(audio_path),
            "-t", f"{dur:.3f}",
            "-ar", "16000", "-ac", "1",
            "-f", "s16le", "-"
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        samples = np.frombuffer(proc.stdout, dtype=np.int16)
        if len(samples) < 2400:
            return "male"
        zero_crossings = np.nonzero(np.diff(samples > 0))[0]
        approx_freq = (len(zero_crossings) / (len(samples) / 16000.0)) / 2.0
        return "female" if approx_freq >= 168.0 else "male"
    except Exception:
        return "male"

def normalize_voice_role(role: str) -> str:
    val = str(role or "").strip().lower()
    if val in {"female", "ស្រី", "sreymom", "woman", "girl", "f"}:
        return "female"
    return "male"

def generate_voice_raw(text: str, role: str, out_wav_path: str, kiri_key: str = "") -> bool:
    clean_text = str(text or "").strip() or "បាទ"
    temp_audio = Path(out_wav_path).with_suffix(".mp3")
    is_female = (normalize_voice_role(role) == "female")

    if kiri_key:
        try:
            url = sanitize_url("https://api.kiritts.com/v1/audio/speech")
            voice_id = "km-female-natural" if is_female else "km-male-natural"
            headers = {"Authorization": f"Bearer {kiri_key}", "Content-Type": "application/json"}
            payload = {"text": clean_text, "voice": voice_id, "response_format": "mp3", "speed": 1.0}
            resp = HTTP_SESSION.post(url, json=payload, headers=headers, timeout=5)
            if resp.status_code == 200:
                with open(str(temp_audio), "wb") as f:
                    f.write(resp.content)
        except Exception:
            pass

    if not temp_audio.exists() or temp_audio.stat().st_size < 100:
        voice_name = "km-KH-SreymomNeural" if is_female else "km-KH-PisethNeural"
        try:
            cmd_tts = [sys.executable, "-m", "edge_tts", "--voice", voice_name, "--text", clean_text, "--write-media", str(temp_audio)]
            subprocess.run(cmd_tts, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=7)
        except Exception:
            pass

    if not temp_audio.exists() or temp_audio.stat().st_size < 100:
        return False

    res = subprocess.run([
        FFMPEG_BIN, "-y", "-i", str(temp_audio),
        "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(out_wav_path)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    temp_audio.unlink(missing_ok=True)
    return res.returncode == 0 and os.path.exists(out_wav_path)

def fit_audio_exact_to_scene(raw_wav: str, scene_dur: float, out_wav: str):
    dur, _, _ = get_media_duration_and_size(raw_wav)
    if dur <= 0.05 or scene_dur <= 0.05:
        shutil.copyfile(raw_wav, out_wav)
        return
    target_dur = max(0.35, scene_dur - 0.04)
    speed_ratio = max(0.85, min(dur / target_dur, 1.55))
    cmd = [
        FFMPEG_BIN, "-y", "-i", str(raw_wav),
        "-filter:a", f"atempo={speed_ratio:.3f}",
        "-t", f"{scene_dur:.3f}",
        "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(out_wav)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.exists(out_wav):
        shutil.copyfile(raw_wav, out_wav)

def split_segment_into_sentences(segment: dict) -> List[dict]:
    raw_text = str(segment["text"]).strip()
    start_t, end_t = float(segment["start"]), float(segment["end"])
    total_dur = max(0.2, end_t - start_t)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?。！？；;៖៕])\s*", raw_text) if s.strip()]
    if len(sentences) <= 1:
        return [segment]
    weights = [max(1, len(re.findall(r"\S+", s))) for s in sentences]
    total_w = sum(weights) or 1
    res, curr_t = [], start_t
    for idx, s in enumerate(sentences):
        seg_dur = total_dur * (weights[idx] / total_w)
        next_t = round(curr_t + seg_dur, 2) if idx < len(sentences) - 1 else end_t
        res.append({"start": round(curr_t, 2), "end": next_t, "text": s})
        curr_t = next_t
    return res

@app.post("/api/submit-payment-receipt")
async def submit_payment_receipt(
    receipt: UploadFile = File(...),
    service_key: str = Form(...),
    plan_type: str = Form(...),
    session_id: str = Form(""),
    sid: Optional[str] = Cookie(None)
):
    active_sid = session_id or sid or "default_user"
    user_dir, uploads_dir, _, _ = get_session_workspace(active_sid)
    ext = Path(receipt.filename).suffix.lower() or ".jpg"
    receipt_dest = uploads_dir / f"receipt_{uuid.uuid4().hex[:6]}{ext}"
    with open(receipt_dest, "wb") as buffer:
        shutil.copyfileobj(receipt.file, buffer)
    req_id = uuid.uuid4().hex[:8]
    prices = {"splitter_month": "1.50$", "splitter_lifetime": "15.00$", "studio_month": "5.00$", "studio_lifetime": "49.99$"}
    price = prices.get(f"{service_key}_{plan_type}", "N/A")
    s_names = {"splitter": "Tool កាត់បំបែកភាគ", "studio": "KHMERDUB Studio Pro"}
    db = load_licenses()
    db.setdefault("requests", {})[req_id] = {
        "client_sid": active_sid, "service_key": service_key, "plan_type": plan_type,
        "receipt_file": str(receipt_dest), "status": "pending", "created_at": time.time()
    }
    save_licenses(db)
    send_telegram_receipt_photo(req_id, active_sid, s_names.get(service_key, service_key), plan_type, price, receipt_dest)
    return {"status": "submitted", "request_id": req_id}

@app.get("/api/check-subscription-status")
async def check_subscription_status(service_key: str, request_id: Optional[str] = None, session_id: str = "", sid: Optional[str] = Cookie(None)):
    active_sid = session_id or sid or "default_user"
    db = load_licenses()
    
    client_sub = db.get("clients", {}).get(active_sid, {}).get(service_key)
    if client_sub:
        if client_sub.get("plan") == "lifetime":
            return {"active": True, "type": "lifetime", "days_left": "មួយជីវិត"}
        if client_sub.get("plan") == "month":
            exp = client_sub.get("expires_at", 0)
            now = time.time()
            if now < exp:
                return {"active": True, "type": "month", "days_left": f"{max(1, int((exp - now)/86400))} ថ្ងៃ"}
            return {"active": False, "reason": "expired"}

    req_status = "none"
    if request_id and request_id in db.get("requests", {}):
        req_status = db["requests"][request_id].get("status", "pending")
        if req_status == "approved":
            return {"active": True, "type": db["requests"][request_id].get("plan_type", "month")}

    return {"active": False, "request_status": req_status}

@app.post("/api/transcribe-episode")
async def transcribe_episode(
    episode_id: str = Form(...),
    session_id: str = Form(""),
    model_size: str = Form("gemini"),
    cloud_key: str = Form(""),
    lang: str = Form("auto"),
    speaker_mode: str = Form("auto"),
    sid: Optional[str] = Cookie(None)
):
    active_sid = session_id or sid or "default_user"
    _, uploads_dir, _, db_file = get_session_workspace(active_sid)
    user_db = load_user_db(db_file)
    if episode_id not in user_db:
        raise HTTPException(status_code=404, detail="Episode not found")

    user_db[episode_id]["status"] = "transcribing"
    user_db[episode_id]["progress"] = 15
    save_user_db(db_file, user_db)

    ep = user_db[episode_id]
    audio_path = uploads_dir / f"{episode_id}_audio.wav"
    subprocess.run([FFMPEG_BIN, "-y", "-i", str(ep["path"]), "-vn", "-ar", "16000", "-ac", "1", str(audio_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    total_dur, _, _ = get_media_duration_and_size(str(ep["path"]))
    total_dur = total_dur or 60.0

    raw_segments = []
    if WhisperModel:
        try:
            whisper = WhisperModel("tiny", device="cpu", compute_type="int8", cpu_threads=4)
            segments, _ = whisper.transcribe(str(audio_path), vad_filter=True, beam_size=1, language=None if lang == "auto" else lang)
            for seg in segments:
                if seg.text.strip():
                    raw_segments.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "text": seg.text.strip()})
        except Exception:
            pass

    if not raw_segments or len(raw_segments) <= 1:
        raw_segments = []
        slice_dur, curr_t = 4.5, 0.0
        while curr_t < total_dur:
            next_t = round(min(total_dur, curr_t + slice_dur), 2)
            raw_segments.append({"start": round(curr_t, 2), "end": next_t, "text": f"Line {int(curr_t)}s"})
            curr_t = next_t

    user_db[episode_id]["progress"] = 40
    save_user_db(db_file, user_db)

    split_segments = []
    for seg in raw_segments:
        split_segments.extend(split_segment_into_sentences(seg))
    for idx, item in enumerate(split_segments):
        item["id"] = idx + 1

    ai_map = {}
    clean_k = cloud_key.strip()
    if clean_k and model_size in ("gemini", "mwapi"):
        chunk_size = 10
        chunks = [split_segments[i:i + chunk_size] for i in range(0, len(split_segments), chunk_size)]
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(call_ai_chunk, c, clean_k, model_size, lang) for c in chunks]
            for f in futures:
                try:
                    for r in f.result():
                        if isinstance(r, dict) and "id" in r:
                            ai_map[int(r["id"])] = r
                except Exception:
                    pass

    user_db[episode_id]["progress"] = 70
    save_user_db(db_file, user_db)

    needed_fallback = []
    for orig in split_segments:
        seg_id = orig["id"]
        t_val = str(ai_map.get(seg_id, {}).get("khmer_text", "")).strip()
        if not re.search(r"[\u1780-\u17ff]", t_val) or t_val == orig["text"]:
            needed_fallback.append(orig)

    if needed_fallback:
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_orig = {executor.submit(robust_translate_to_khmer, o["text"]): o for o in needed_fallback}
            for fut in as_completed(future_to_orig):
                o = future_to_orig[fut]
                seg_id = o["id"]
                try:
                    km_res = fut.result()
                    if km_res and re.search(r"[\u1780-\u17ff]", km_res):
                        ai_map.setdefault(seg_id, {})["khmer_text"] = km_res
                except Exception:
                    pass

    dialogues = []
    for orig in split_segments:
        seg_id = orig["id"]
        ai_res = ai_map.get(seg_id, {})
        acoustic_gender = detect_speaker_acoustic_pitch(str(audio_path), orig["start"], orig["end"])
        final_role = acoustic_gender if acoustic_gender in ("male", "female") else ai_res.get("role", "female")

        km_text = str(ai_res.get("khmer_text", "")).strip()
        if not re.search(r"[\u1780-\u17ff]", km_text):
            km_text = robust_translate_to_khmer(orig["text"])
        if not km_text:
            km_text = orig["text"]

        forced = normalize_voice_role(speaker_mode)
        if speaker_mode in ("male", "female"):
            final_role = forced

        dialogues.append({
            "id": seg_id,
            "start": orig["start"],
            "end": orig["end"],
            "original_text": orig["text"],
            "khmer_text": km_text,
            "speaker": final_role,
            "speaker_id": str(ai_res.get("speaker_id", f"S{seg_id}")),
            "voice": "Sreymom" if final_role == "female" else "Piseth"
        })

    user_db[episode_id]["dialogues"] = dialogues
    user_db[episode_id]["status"] = "transcribed"
    user_db[episode_id]["progress"] = 100
    save_user_db(db_file, user_db)
    return JSONResponse({"episode_id": episode_id, "dialogues": dialogues})

@app.get("/")
async def root_view(response: Response, sid: Optional[str] = Cookie(None)):
    session_id = sid or uuid.uuid4().hex[:12]
    response.set_cookie(key="sid", value=session_id, max_age=86400 * 30, httponly=False)
    p_html = STATIC_DIR / "index.html"
    return FileResponse(p_html if p_html.exists() else (BASE_DIR / "index.html"))

def stream_file_ranges(file_path: Path, range_header: Optional[str] = None):
    file_size = file_path.stat().st_size
    start = 0
    end = file_size - 1
    if range_header:
        m = re.search(r"bytes=(\d+)-(\d*)", range_header)
        if m:
            start = int(m.group(1))
            if m.group(2):
                end = int(m.group(2))
    start = max(0, min(start, file_size - 1))
    end = max(start, min(end, file_size - 1))
    content_length = end - start + 1

    def iterfile():
        with open(file_path, "rb") as f:
            f.seek(start)
            rem = content_length
            while rem > 0:
                c = min(rem, 1024 * 512)
                d = f.read(c)
                if not d:
                    break
                rem -= len(d)
                yield d

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(content_length),
        "Content-Type": "video/mp4"
    }
    return StreamingResponse(iterfile(), status_code=206 if range_header else 200, headers=headers)

@app.get("/api/video/{session_id}/{ep_id}")
async def stream_video(session_id: str, ep_id: str, request: Request, sid: Optional[str] = Cookie(None)):
    active_sid = session_id or sid or "default_user"
    _, uploads_dir, processed_dir, db_file = get_session_workspace(active_sid)
    user_db = load_user_db(db_file)
    if ep_id not in user_db:
        _, uploads_dir, processed_dir, db_file = get_session_workspace("default_user")
        user_db = load_user_db(db_file)
    if ep_id not in user_db:
        raise HTTPException(status_code=404, detail="Video not found")
    ep = user_db[ep_id]
    target_path = Path(ep["path"])
    if ep.get("status") == "completed" and ep.get("output_file"):
        dubbed_file = processed_dir / ep["output_file"]
        if dubbed_file.exists():
            target_path = dubbed_file

    if not target_path.exists():
        matches = list(uploads_dir.glob(f"{ep_id}*"))
        if matches:
            target_path = matches[0]
        else:
            raise HTTPException(status_code=404, detail="File missing on disk")

    range_header = request.headers.get("range")
    return stream_file_ranges(target_path, range_header)

@app.post("/api/split-video")
async def split_video_api(file: UploadFile = File(...), custom_title: str = Form(""), split_minutes: float = Form(3.0), action_type: str = Form("dubbing"), session_id: str = Form(""), sid: Optional[str] = Cookie(None)):
    active_sid = session_id or sid or "default_user"
    _, uploads_dir, _, db_file = get_session_workspace(active_sid)
    user_db = load_user_db(db_file)
    temp_input = uploads_dir / f"src_{uuid.uuid4().hex[:6]}_{file.filename}"
    with open(temp_input, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    total_dur, _, _ = get_media_duration_and_size(str(temp_input))
    part_seconds = max(15.0, float(split_minutes) * 60.0)
    current_count = len(user_db)
    generated = []
    base_name = custom_title.strip() or Path(file.filename).stem

    part_idx, start_t = 1, 0.0
    while start_t < total_dur:
        ep_id = f"ep_{uuid.uuid4().hex[:6]}"
        ep_code = f"EP{current_count + part_idx:02d}"
        out_path = uploads_dir / f"{ep_id}_{ep_code}.mp4"
        cmd = [FFMPEG_BIN, "-y", "-ss", f"{start_t:.3f}", "-i", str(temp_input), "-t", f"{part_seconds:.3f}", "-c", "copy", "-avoid_negative_ts", "make_zero", str(out_path)]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if out_path.exists():
            ep_item = {
                "id": ep_id, "code": ep_code, "filename": f"{base_name}_{ep_code}.mp4", "path": str(out_path),
                "download_url": f"/workspace/{active_sid}/uploads/{ep_id}_{ep_code}.mp4",
                "video_url": f"/api/video/{active_sid}/{ep_id}", "status": "ready", "progress": 0, "dialogues": []
            }
            if action_type == "dubbing":
                user_db[ep_id] = ep_item
            generated.append(ep_item)
            part_idx += 1
        start_t += part_seconds
    temp_input.unlink(missing_ok=True)
    if action_type == "dubbing":
        save_user_db(db_file, user_db)
    return {"status": "success", "parts_count": len(generated), "parts": generated}

@app.post("/api/batch-upload")
async def batch_upload(files: List[UploadFile] = File(...), session_id: str = Form(""), sid: Optional[str] = Cookie(None)):
    active_sid = session_id or sid or "default_user"
    _, uploads_dir, _, db_file = get_session_workspace(active_sid)
    user_db = load_user_db(db_file)
    uploaded = []
    current_count = len(user_db)
    for i, f in enumerate(files):
        ep_id = f"ep_{uuid.uuid4().hex[:6]}"
        ep_code = f"EP{current_count + i + 1:02d}"
        clean_fn = Path(f.filename).stem
        dest_path = uploads_dir / f"{ep_id}_{ep_code}.mp4"
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(f.file, buffer)
        user_db[ep_id] = {
            "id": ep_id, "code": ep_code, "filename": clean_fn, "path": str(dest_path),
            "video_url": f"/api/video/{active_sid}/{ep_id}", "status": "ready", "progress": 0, "dialogues": []
        }
        uploaded.append(user_db[ep_id])
    save_user_db(db_file, user_db)
    return {"uploaded": uploaded}

@app.get("/api/episodes")
async def get_episodes(session_id: str = "", sid: Optional[str] = Cookie(None)):
    active_sid = session_id or sid or "default_user"
    _, _, _, db_file = get_session_workspace(active_sid)
    return JSONResponse(load_user_db(db_file))

@app.delete("/api/episodes/{episode_id}")
async def delete_single_episode(episode_id: str, session_id: str = "", sid: Optional[str] = Cookie(None)):
    active_sid = session_id or sid or "default_user"
    _, uploads_dir, processed_dir, db_file = get_session_workspace(active_sid)
    user_db = load_user_db(db_file)
    if episode_id in user_db:
        del user_db[episode_id]
        save_user_db(db_file, user_db)
        return {"status": "success"}
    return JSONResponse(status_code=404, content={"message": "Not found"})

@app.post("/api/clear-all-episodes")
async def clear_all_episodes(session_id: str = Form(""), sid: Optional[str] = Cookie(None)):
    active_sid = session_id or sid or "default_user"
    user_dir, uploads_dir, processed_dir, db_file = get_session_workspace(active_sid)
    shutil.rmtree(user_dir, ignore_errors=True)
    user_dir.mkdir(parents=True, exist_ok=True)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    save_user_db(db_file, {})
    return {"status": "success"}

@app.post("/api/update-dialogues")
async def update_dialogues_api(episode_id: str = Form(...), dialogues_json: str = Form(...), session_id: str = Form(""), sid: Optional[str] = Cookie(None)):
    active_sid = session_id or sid or "default_user"
    _, _, _, db_file = get_session_workspace(active_sid)
    user_db = load_user_db(db_file)
    if episode_id in user_db:
        user_db[episode_id]["dialogues"] = json.loads(dialogues_json)
        save_user_db(db_file, user_db)
        return {"status": "success"}
    raise HTTPException(status_code=404, detail="Not found")

def process_single_dialogue_voice(it, idx, ep_id, processed_dir, kiri_key):
    txt = str(it.get("khmer_text", "")).strip()
    if not txt:
        return None
    r_wav = processed_dir / f"{ep_id}_r_{idx}.wav"
    f_wav = processed_dir / f"{ep_id}_f_{idx}.wav"
    if generate_voice_raw(txt, it.get("speaker", "male"), str(r_wav), kiri_key):
        fit_audio_exact_to_scene(str(r_wav), float(it["end"]) - float(it["start"]), str(f_wav))
        r_wav.unlink(missing_ok=True)
        return {"start": float(it["start"]), "path": str(f_wav)}
    return None

def make_even(n: int) -> int:
    n = int(round(n))
    return n if n % 2 == 0 else n + 1

def render_worker(active_sid: str, cfg_dict: dict):
    with RENDER_LOCK:
        _, uploads_dir, processed_dir, db_file = get_session_workspace(active_sid)
        ep_ids = cfg_dict.get("episode_ids", [])
        kiri_key = cfg_dict.get("kiri_api_key", "")
        keep_bgm = bool(cfg_dict.get("keep_bgm", True))
        encoder_choice = get_best_video_encoder()

        user_db = load_user_db(db_file)
        for ep_id in ep_ids:
            if ep_id in user_db and user_db[ep_id].get("status") != "completed":
                user_db[ep_id]["status"] = "waiting"
                user_db[ep_id]["progress"] = 0
        save_user_db(db_file, user_db)

        for ep_id in ep_ids:
            user_db = load_user_db(db_file)
            if ep_id not in user_db:
                continue

            user_db[ep_id]["status"] = "rendering"
            user_db[ep_id]["progress"] = 10
            save_user_db(db_file, user_db)

            ep = user_db[ep_id]
            video_in = ep["path"]
            total_dur, base_w, base_h = get_media_duration_and_size(video_in)
            total_dur = total_dur or 60.0

            dialogues = ep.get("dialogues", [])
            clean_bgm_wav = processed_dir / f"{ep_id}_clean_bgm.wav"
            if keep_bgm:
                separate_and_purge_original_vocals(str(video_in), clean_bgm_wav)

            clips = []
            if dialogues:
                with ThreadPoolExecutor(max_workers=8) as executor:
                    futures = [executor.submit(process_single_dialogue_voice, it, idx, ep_id, processed_dir, kiri_key) for idx, it in enumerate(dialogues)]
                    for f in as_completed(futures):
                        res = f.result()
                        if res:
                            clips.append(res)
                clips.sort(key=lambda x: x["start"])

            dub_track = processed_dir / f"{ep_id}_dub.wav"
            if clips:
                inputs_cmd = ["-f", "lavfi", "-t", f"{total_dur:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
                filter_parts = []
                for idx, c in enumerate(clips):
                    inputs_cmd.extend(["-i", str(c["path"])])
                    ms = int(max(0, float(c["start"])) * 1000)
                    filter_parts.append(f"[{idx+1}:a]adelay={ms}|{ms}[a{idx+1}]")
                filter_complex = ";".join(filter_parts) + ";[0:a]" + "".join([f"[a{i+1}]" for i in range(len(clips))]) + f"amix=inputs={len(clips)+1}:dropout_transition=0:normalize=0[aout]"
                subprocess.run([FFMPEG_BIN, "-y", *inputs_cmd, "-filter_complex", filter_complex, "-map", "[aout]", "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(dub_track)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                subprocess.run([FFMPEG_BIN, "-y", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", f"{total_dur:.3f}", "-c:a", "pcm_s16le", str(dub_track)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            user_db = load_user_db(db_file)
            if ep_id in user_db:
                user_db[ep_id]["progress"] = 40
                save_user_db(db_file, user_db)

            v_nodes = []
            curr_v = "[0:v]"

            blur_items = cfg_dict.get("blur_items", []) or []
            for b_idx, b in enumerate(blur_items):
                raw_bx = float(b.get("norm_x", 0.10)) * base_w
                raw_by = float(b.get("norm_y", 0.80)) * base_h
                raw_bw = float(b.get("norm_w", 0.80)) * base_w
                raw_bh = float(b.get("norm_h", 0.12)) * base_h

                bx = make_even(max(0, min(raw_bx, base_w - 20)))
                by = make_even(max(0, min(raw_by, base_h - 20)))
                bw = make_even(max(20, min(raw_bw, base_w - bx)))
                bh = make_even(max(10, min(raw_bh, base_h - by)))

                v_nodes.append(f"{curr_v}split=2[main_{b_idx}][crop_src_{b_idx}]")
                v_nodes.append(f"[crop_src_{b_idx}]crop={bw}:{bh}:{bx}:{by},boxblur=15:3[patch_{b_idx}]")
                v_nodes.append(f"[main_{b_idx}][patch_{b_idx}]overlay={bx}:{by}[v_blurred_{b_idx}]")
                curr_v = f"[v_blurred_{b_idx}]"

            text_items = cfg_dict.get("text_items", []) or []
            for t_idx, t in enumerate(text_items):
                raw_txt = str(t.get("text", "")).strip()
                if raw_txt:
                    clean_txt = raw_txt.replace(":", "\\:").replace("'", "").replace("%", "")
                    tx = max(0, min(int(float(t.get("norm_x", 0.05)) * base_w), base_w - 50))
                    ty = max(0, min(int(float(t.get("norm_y", 0.05)) * base_h), base_h - 30))
                    tsize = int(t.get("size", 34))
                    next_v = f"[v_txt_{t_idx}]"
                    v_nodes.append(f"{curr_v}drawtext=text='{clean_txt}':fontcolor=gold:fontsize={tsize}:x={tx}:y={ty}:shadowcolor=black@0.9:shadowx=2:shadowy=2{next_v}")
                    curr_v = next_v

            if base_h > base_w:
                target_4k_w, target_4k_h = 2160, 3840
            else:
                target_4k_w, target_4k_h = 3840, 2160

            v_nodes.append(f"{curr_v}scale={target_4k_w}:{target_4k_h}:flags=lanczos,unsharp=5:5:1.0:5:5:0.0[v_final]")
            video_filter_complex = ";".join(v_nodes)

            dub_vol = float(cfg_dict.get("dub_volume", 2.20))
            raw_title = ep.get("filename", "") or "Part"
            clean_title = re.sub(r"[^\w\-_]", "", raw_title) or "Dubbed"
            out_name = f"{ep.get('code', 'EP')}_{clean_title}_4K_Dubbed.mp4"
            final_mp4 = processed_dir / out_name

            if keep_bgm and clean_bgm_wav.exists():
                audio_filter = f"[1:a]volume=0.90[bgm];[2:a]volume={dub_vol}[dub];[bgm][dub]amix=inputs=2:duration=first:dropout_transition=0[a_final]"
                audio_inputs = ["-i", str(clean_bgm_wav), "-i", str(dub_track)]
            else:
                audio_filter = f"[1:a]volume={dub_vol}[a_final]"
                audio_inputs = ["-i", str(dub_track)]

            full_filter_complex = f"{audio_filter};{video_filter_complex}"

            if encoder_choice == "h264_nvenc":
                v_codec_args = [
                    "-c:v", "h264_nvenc",
                    "-preset", "p5",
                    "-rc", "vbr",
                    "-cq", "14",
                    "-b:v", "28M",
                    "-maxrate", "35M",
                    "-bufsize", "40M",
                    "-pix_fmt", "yuv420p"
                ]
            else:
                v_codec_args = [
                    "-c:v", "libx264",
                    "-preset", "faster",
                    "-crf", "14",
                    "-b:v", "24M",
                    "-maxrate", "30M",
                    "-bufsize", "35M",
                    "-pix_fmt", "yuv420p",
                    "-threads", "0"
                ]

            render_cmd = [
                FFMPEG_BIN, "-y",
                "-i", str(video_in),
                *audio_inputs,
                "-filter_complex", full_filter_complex,
                "-map", "[v_final]", "-map", "[a_final]",
                *v_codec_args,
                "-c:a", "aac", "-b:a", "320k",
                "-movflags", "+faststart",
                "-t", f"{total_dur:.3f}",
                str(final_mp4)
            ]

            proc = subprocess.run(render_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode != 0:
                print(f"[FFmpeg Detailed Error]: {proc.stderr[:500]}")

            for c in clips:
                Path(c["path"]).unlink(missing_ok=True)
            dub_track.unlink(missing_ok=True)
            if clean_bgm_wav.exists():
                clean_bgm_wav.unlink(missing_ok=True)

            user_db = load_user_db(db_file)
            if ep_id in user_db and final_mp4.exists() and final_mp4.stat().st_size > 1000:
                user_db[ep_id]["status"] = "completed"
                user_db[ep_id]["progress"] = 100
                user_db[ep_id]["output_file"] = out_name
                user_db[ep_id]["output_url"] = f"/api/download/{active_sid}/{out_name}"
                save_user_db(db_file, user_db)

@app.post("/api/batch-render")
async def start_batch_render(cfg: BatchRenderConfig, sid: Optional[str] = Cookie(None)):
    active_sid = cfg.session_id or sid or "default_user"
    threading.Thread(target=render_worker, args=(active_sid, cfg.dict()), daemon=True).start()
    return JSONResponse({"message": "Auto 4K Render started"})

@app.get("/api/download/{session_id}/{filename}")
async def download_user_file(session_id: str, filename: str):
    _, _, processed_dir, _ = get_session_workspace(session_id)
    target = processed_dir / filename
    if not target.exists():
        matches = list(processed_dir.glob(f"*{filename}*"))
        if matches:
            target = matches[0]
        else:
            raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=str(target), filename=filename, media_type="video/mp4")

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 50)
    print(" 🚀 KHMERDUB STUDIO PRO - AUTO 4K & TELEGRAM SYNC ENGINE")
    print(" 👉 http://127.0.0.1:8080")
    print("=" * 50 + "\n")
    uvicorn.run("server:app", host="0.0.0.0", port=8080, log_level="info")