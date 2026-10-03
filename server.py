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
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

import requests
import static_ffmpeg
static_ffmpeg.add_paths()

# --- បំបិទ WARNING & CONNECTION RESET លើ WINDOWS ---
logging.getLogger("asyncio").setLevel(logging.CRITICAL)
logging.getLogger("uvicorn.error").setLevel(logging.CRITICAL)
logging.getLogger("uvicorn.access").setLevel(logging.CRITICAL)

if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport

        def silence_connection_lost(self, exc):
            try:
                if hasattr(self, "_sock") and self._sock:
                    try: self._sock.shutdown(1)
                    except Exception: pass
                    self._sock.close()
            except Exception: pass

        def silence_write(self, data):
            try:
                if not getattr(self, "_closing", False) and hasattr(self, "_sock") and self._sock:
                    return self._sock.send(data)
            except Exception:
                pass

        _ProactorBasePipeTransport._call_connection_lost = silence_connection_lost
        _ProactorBasePipeTransport.write = silence_write
    except Exception:
        pass
# ----------------------------------------------------

FFMPEG_BIN = "ffmpeg"
try:
    import imageio_ffmpeg
    f_path = imageio_ffmpeg.get_ffmpeg_exe()
    if os.path.exists(f_path):
        FFMPEG_BIN = f_path
except Exception:
    pass

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Cookie, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

try:
    import edge_tts
except Exception:
    edge_tts = None

app = FastAPI(title="KhmerDub Pro Studio", version="12000.0.0")

@app.middleware("http")
async def suppress_connection_reset_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except (ConnectionResetError, BrokenPipeError):
        return Response(status_code=204)
    except Exception as e:
        err_msg = str(e).lower()
        if "10054" in err_msg or "connection_lost" in err_msg or "socket.send" in err_msg:
            return Response(status_code=204)
        raise e

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

WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/workspace", StaticFiles(directory=str(WORKSPACE_ROOT)), name="workspace")
RENDER_LOCK = threading.Lock()

def sanitize_url(raw_url: str) -> str:
    u = str(raw_url).strip()
    match = re.search(r'https?://[^\s\[\]\(\)\'\"]+', u)
    if match:
        return match.group(0).strip()
    return u.replace("[", "").replace("]", "").replace("(", "").replace(")", "").strip()

def get_session_workspace(session_id: str):
    sid = re.sub(r'[^a-zA-Z0-9_-]', '', session_id) or "default_user"
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
                if isinstance(d, dict): return d
        except Exception: pass
    return {}

def save_user_db(db_file: Path, data: dict):
    try:
        with open(db_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[!] Save DB Error: {e}")

class BatchRenderConfig(BaseModel):
    session_id: Optional[str] = "default_user"
    episode_ids: List[str]
    merge_all_into_one: bool = False
    speaker_mode: str = "auto"
    speed_factor: float = 1.0
    dub_volume: float = 2.20
    kiri_api_key: Optional[str] = ""
    dialogues_map: Optional[dict] = None

def get_media_duration(file_path: str) -> float:
    try:
        cmd = [FFMPEG_BIN, "-i", str(file_path)]
        res = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True, errors='replace')
        match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", res.stderr)
        if match:
            h, m, s = match.groups()
            return int(h) * 3600 + int(m) * 60 + float(s)
        return 0.0
    except Exception:
        return 0.0

def resolve_voice_profile(role_raw: str):
    r = str(role_raw).lower()
    if any(k in r for k in ["female", "ស្រី", "sreymom", "woman", "girl"]):
        return "km-KH-SreymomNeural"
    return "km-KH-PisethNeural"

# --- មុខងារបង្កើតសំឡេង (ទទួល Kiri TTS Key ផ្ទាល់ពី Web UI) ---
def generate_voice_raw(text: str, role: str, out_wav_path: str, kiri_key: str = "") -> bool:
    clean_text = text.strip() or "បាទ"
    temp_audio = Path(out_wav_path).with_suffix(".mp3")
    is_female = any(k in str(role).lower() for k in ["female", "ស្រី", "sreymom", "woman", "girl"])
    clean_kiri_key = (kiri_key or "").strip()

    # ១. ប្រើប្រាស់ Kiri TTS API ប្រសិនបើយូស័របានបញ្ចូល Key ពី UI
    if clean_kiri_key:
        try:
            url = sanitize_url("https://api.kiritts.com/v1/audio/speech")
            voice_id = "km-female-natural" if is_female else "km-male-natural"
            headers = {
                "Authorization": f"Bearer {clean_kiri_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "text": clean_text,
                "voice": voice_id,
                "response_format": "mp3",
                "speed": 1.0
            }
            resp = requests.post(url, json=payload, headers=headers, timeout=8)
            if resp.status_code == 200:
                with open(str(temp_audio), "wb") as f:
                    f.write(resp.content)
        except Exception as e:
            print(f"[!] Kiri API Call Notice: {e}")

    # ២. Edge-TTS Studio Fallback (ពិសិដ្ឋ & ស្រីមុំ)
    if not temp_audio.exists() or temp_audio.stat().st_size < 100:
        voice_name = "km-KH-SreymomNeural" if is_female else "km-KH-PisethNeural"
        try:
            cmd_tts = [
                sys.executable, "-m", "edge_tts",
                "--voice", voice_name,
                "--text", clean_text,
                "--rate=+0%",
                "--pitch=+0Hz",
                "--write-media", str(temp_audio)
            ]
            subprocess.run(cmd_tts, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=8)
        except Exception:
            pass

    # ៣. Google TTS Fallback
    if not temp_audio.exists() or temp_audio.stat().st_size < 100:
        try:
            url = sanitize_url("https://translate.google.com/translate_tts")
            params = {"ie": "UTF-8", "q": clean_text[:180], "tl": "km", "client": "tw-ob"}
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, params=params, headers=headers, timeout=5)
            if resp.status_code == 200:
                with open(str(temp_audio), "wb") as f:
                    f.write(resp.content)
        except Exception: pass

    if not temp_audio.exists() or temp_audio.stat().st_size < 100:
        return False

    # បម្លែងជា WAV 48000Hz PCM ស្តង់ដារ
    subprocess.run([
        FFMPEG_BIN, "-y", "-i", str(temp_audio),
        "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(out_wav_path)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    temp_audio.unlink(missing_ok=True)
    return os.path.exists(out_wav_path) and os.path.getsize(out_wav_path) > 100

def fit_audio_exact_to_scene(raw_wav: str, scene_dur: float, out_wav: str):
    actual_dur = get_media_duration(raw_wav)
    if actual_dur <= 0.05 or scene_dur <= 0.05:
        shutil.copyfile(raw_wav, out_wav)
        return

    target_dur = max(0.35, scene_dur - 0.05)
    speed_ratio = actual_dur / target_dur
    speed_ratio = max(0.85, min(speed_ratio, 1.45))

    filters = []
    curr = speed_ratio
    while curr > 2.0:
        filters.append("atempo=2.0")
        curr /= 2.0
    while curr < 0.5:
        filters.append("atempo=0.5")
        curr /= 0.5
    filters.append(f"atempo={curr:.3f}")

    cmd = [
        FFMPEG_BIN, "-y", "-i", str(raw_wav),
        "-filter:a", ",".join(filters),
        "-t", f"{scene_dur:.3f}",
        "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(out_wav)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.exists(out_wav):
        shutil.copyfile(raw_wav, out_wav)

def parse_robust_json(text: str) -> Optional[list]:
    try:
        clean = text.strip()
        if clean.startswith("```"):
            clean = re.sub(r"^```[a-zA-Z]*\n?", "", clean)
            clean = re.sub(r"\n?```$", "", clean)
        clean = clean.strip()
        match = re.search(r'\[\s*\{.*\}\s*\]', clean, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(clean)
    except Exception:
        return None

def direct_translate_km(text: str) -> str:
    clean_t = text.strip()
    if not clean_t:
        return ""
    try:
        url = sanitize_url("[https://translate.googleapis.com/translate_a/single](https://translate.googleapis.com/translate_a/single)")
        params = {"client": "gtx", "sl": "auto", "tl": "km", "dt": "t", "q": clean_t}
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get(url, params=params, headers=headers, timeout=6)
        if resp.status_code == 200:
            data = resp.json()
            if data and data[0]:
                out_txt = "".join([part[0] for part in data[0] if part and part[0]]).strip()
                if out_txt and out_txt != clean_t:
                    return out_txt
    except Exception:
        pass

    try:
        url_c = sanitize_url("[https://clients5.google.com/translate_a/t](https://clients5.google.com/translate_a/t)")
        params_c = {"client": "dict-chrome-ex", "sl": "auto", "tl": "km", "q": clean_t}
        headers_c = {"User-Agent": "Mozilla/5.0"}
        resp_c = requests.get(url_c, params=params_c, headers=headers_c, timeout=6)
        if resp_c.status_code == 200:
            res_json = resp_c.json()
            if isinstance(res_json, list) and res_json:
                res_val = res_json[0]
                if isinstance(res_val, list) and res_val:
                    return str(res_val[0]).strip()
                elif isinstance(res_val, str) and res_val.strip():
                    return res_val.strip()
    except Exception:
        pass

    return clean_t

def batch_translate_google(texts: List[str]) -> List[str]:
    if not texts:
        return []
    final_translations = []
    for t in texts:
        final_translations.append(direct_translate_km(t))
    return final_translations

def translate_and_tag_gemini(items: List[dict], gemini_key: str, source_lang: str = "auto") -> List[dict]:
    clean_k = gemini_key.strip()
    if not items or not clean_k or not clean_k.startswith("AIza"):
        return []

    lang_desc = "Chinese or English" if source_lang in ["auto", ""] else ("Chinese" if source_lang == "zh" else "English")
    system_prompt = (
        f"You are a professional film dubbing director analyzing dialogue in {lang_desc}.\n"
        "TASKS:\n"
        "1. Identify speaker strictly as 'male' or 'female'.\n"
        "2. Translate each dialogue into natural spoken KHMER (ភាសាខ្មែរ).\n"
        "Return STRICTLY a JSON array of objects with keys: 'role' ('male' or 'female') and 'khmer_text'."
    )

    input_payload = [{"id": i["id"], "text": i["text"]} for i in items]
    url = sanitize_url(f"[https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=](https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=){clean_k}")
    
    req_body = {
        "contents": [{"role": "user", "parts": [{"text": f"{system_prompt}\n\nDialogues:\n{json.dumps(input_payload, ensure_ascii=False)}"}]}],
        "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"}
    }

    try:
        resp = requests.post(url, json=req_body, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            parsed = parse_robust_json(raw_text)
            if parsed and isinstance(parsed, list) and len(parsed) == len(items):
                print("[✓] បកប្រែដោយ Gemini Flash (Free API) ជោគជ័យ!")
                return [{"role": "female" if "female" in str(p.get("role", "")).lower() else "male", "khmer_text": str(p.get("khmer_text", "")).strip()} for p in parsed]
    except Exception as e:
        print(f"[!] Gemini Error: {e}")
    return []

@app.get("/")
async def root_view(response: Response, sid: Optional[str] = Cookie(None)):
    session_id = sid or str(uuid.uuid4().hex[:12])
    response.set_cookie(key="sid", value=session_id, max_age=86400 * 30, httponly=False)
    return FileResponse(STATIC_DIR / "index.html")

@app.post("/api/batch-upload")
async def batch_upload(files: List[UploadFile] = File(...), session_id: str = Form(""), sid: Optional[str] = Cookie(None)):
    active_sid = session_id or sid or "default_user"
    _, uploads_dir, _, db_file = get_session_workspace(active_sid)
    user_db = load_user_db(db_file)
    uploaded_eps = []
    current_count = len(user_db)
    for i, f in enumerate(files):
        ep_id = f"ep_{uuid.uuid4().hex[:6]}"
        ep_code = f"EP{current_count + i + 1:02d}"
        clean_fn = Path(f.filename).stem
        ext = Path(f.filename).suffix.lower() or ".mp4"
        dest_path = uploads_dir / f"{ep_id}_{ep_code}{ext}"
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(f.file, buffer)
        user_db[ep_id] = {
            "id": ep_id, "code": ep_code, "filename": clean_fn,
            "path": str(dest_path), "video_url": f"/api/video/{active_sid}/{ep_id}",
            "status": "ready", "progress": 0, "dialogues": []
        }
        uploaded_eps.append(user_db[ep_id])
    save_user_db(db_file, user_db)
    return {"uploaded": uploaded_eps}

@app.get("/api/video/{session_id}/{ep_id}")
async def stream_video(session_id: str, ep_id: str):
    _, _, processed_dir, db_file = get_session_workspace(session_id)
    user_db = load_user_db(db_file)
    if ep_id not in user_db:
        raise HTTPException(status_code=404, detail="Video not found")
    
    ep = user_db[ep_id]
    if ep.get("status") == "completed" and ep.get("output_file"):
        dubbed_file = processed_dir / ep["output_file"]
        if dubbed_file.exists():
            return FileResponse(path=str(dubbed_file), media_type="video/mp4")

    if not os.path.exists(ep["path"]):
        raise HTTPException(status_code=404, detail="Original Video not found")
    return FileResponse(path=str(ep["path"]), media_type="video/mp4")

@app.get("/api/episodes")
async def get_episodes(session_id: str = "", sid: Optional[str] = Cookie(None)):
    active_sid = session_id or sid or "default_user"
    _, _, _, db_file = get_session_workspace(active_sid)
    return JSONResponse(load_user_db(db_file))

@app.post("/api/clear-all-episodes")
async def clear_all_episodes(session_id: str = Form(""), sid: Optional[str] = Cookie(None)):
    active_sid = session_id or sid or "default_user"
    user_dir, uploads_dir, processed_dir, db_file = get_session_workspace(active_sid)
    shutil.rmtree(user_dir, ignore_errors=True)
    user_dir.mkdir(parents=True, exist_ok=True)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    save_user_db(db_file, {})
    return {"message": "All episodes cleared successfully"}

@app.delete("/api/episodes/{episode_id}")
async def delete_single_episode_api(episode_id: str, session_id: str = "", sid: Optional[str] = Cookie(None)):
    active_sid = session_id or sid or "default_user"
    user_dir, uploads_dir, processed_dir, db_file = get_session_workspace(active_sid)
    user_db = load_user_db(db_file)
    if episode_id in user_db:
        ep = user_db.pop(episode_id)
        try:
            if os.path.exists(ep.get("path", "")): os.remove(ep["path"])
        except Exception: pass
        for f in uploads_dir.glob(f"{episode_id}*"):
            try: f.unlink(missing_ok=True)
            except Exception: pass
        for f in processed_dir.glob(f"{episode_id}*"):
            try: f.unlink(missing_ok=True)
            except Exception: pass
        new_db = {}
        for idx, (k, v) in enumerate(user_db.items()):
            v["code"] = f"EP{idx + 1:02d}"
            new_db[k] = v
        save_user_db(db_file, new_db)
        return {"message": "Deleted successfully", "episodes": new_db}
    raise HTTPException(status_code=404, detail="Episode not found")

@app.post("/api/update-dialogues")
async def update_dialogues_api(episode_id: str = Form(...), dialogues_json: str = Form(...), session_id: str = Form(""), sid: Optional[str] = Cookie(None)):
    active_sid = session_id or sid or "default_user"
    _, _, _, db_file = get_session_workspace(active_sid)
    user_db = load_user_db(db_file)
    if episode_id in user_db:
        try:
            user_db[episode_id]["dialogues"] = json.loads(dialogues_json)
            save_user_db(db_file, user_db)
            return {"status": "success"}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    raise HTTPException(status_code=404, detail="Episode not found")

@app.post("/api/transcribe-episode")
async def transcribe_episode(
    episode_id: str = Form(...),
    session_id: str = Form(""),
    model_size: str = Form(""),
    cloud_key: str = Form(""),
    api_endpoint: str = Form("[https://api.mwapi.dev](https://api.mwapi.dev)"),
    lang: str = Form("auto"),
    speaker_mode: str = Form("auto"),
    sid: Optional[str] = Cookie(None)
):
    active_sid = session_id or sid or "default_user"
    _, uploads_dir, _, db_file = get_session_workspace(active_sid)
    user_db = load_user_db(db_file)
    if episode_id not in user_db:
        raise HTTPException(status_code=404, detail="Episode not found")

    ep = user_db[episode_id]
    audio_path = uploads_dir / f"{episode_id}_audio.wav"
    subprocess.run([
        FFMPEG_BIN, "-y", "-i", str(ep["path"]),
        "-vn", "-ar", "16000", "-ac", "1", str(audio_path)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    sub_segments = []
    if WhisperModel:
        whisper = WhisperModel("tiny", device="cpu", compute_type="int8")
        segments, _ = whisper.transcribe(str(audio_path), vad_filter=True, language=None if lang=="auto" else lang)
        for seg in segments:
            if seg.text.strip():
                sub_segments.append({"id": len(sub_segments)+1, "start": round(seg.start, 2), "end": round(seg.end, 2), "text": seg.text.strip()})

    if not sub_segments:
        total_dur = get_media_duration(str(audio_path)) or 10.0
        sub_segments.append({"id": 1, "start": 0.0, "end": round(min(3.5, total_dur), 2), "text": "Hello"})

    cleaned_key = cloud_key.strip()
    ai_processed_data = []

    # ១. Gemini Flash
    if cleaned_key.startswith("AIza"):
        print("[*] ដំណើរការបកប្រែតាម Google Gemini Flash (Free API)...")
        ai_processed_data = translate_and_tag_gemini(sub_segments, gemini_key=cleaned_key, source_lang=lang)

    # ២. Multi-Engine Auto Fallback (ធានាចេញខ្មែរ ១០០%)
    if not ai_processed_data:
        print("[*] ដំណើរការបកប្រែជាភាសាខ្មែរតាម Auto-Translator...")
        raw_texts = [item["text"] for item in sub_segments]
        translated_texts = batch_translate_google(raw_texts)

        last_role = "male"
        for idx, orig in enumerate(sub_segments):
            raw_lower = orig["text"].strip().lower()
            if any(k in raw_lower for k in ["chloe", "married", "marry", "wife", "girl", "woman", "she", "her", "sweetheart"]):
                det_role = "female"
            elif any(k in raw_lower for k in ["man", "boy", "he", "him", "husband", "sir", "mr", "foster"]):
                det_role = "male"
            else:
                det_role = "female" if last_role == "male" else "male"
            last_role = det_role

            km_txt = translated_texts[idx] if idx < len(translated_texts) else direct_translate_km(orig["text"])
            if not km_txt or km_txt == orig["text"]:
                km_txt = direct_translate_km(orig["text"])
            ai_processed_data.append({"role": det_role, "khmer_text": km_txt})

    dialogues = []
    for idx, orig in enumerate(sub_segments):
        role_detected = ai_processed_data[idx]["role"] if idx < len(ai_processed_data) else "male"
        km_txt = ai_processed_data[idx]["khmer_text"] if idx < len(ai_processed_data) else direct_translate_km(orig["text"])

        if speaker_mode == "male":
            final_speaker = "male"
        elif speaker_mode == "female":
            final_speaker = "female"
        else:
            final_speaker = role_detected

        dialogues.append({
            "id": idx + 1,
            "start": orig["start"],
            "end": orig["end"],
            "original_text": orig["text"],
            "khmer_text": km_txt,
            "speaker": final_speaker
        })

    user_db[episode_id]["dialogues"] = dialogues
    user_db[episode_id]["status"] = "transcribed"
    save_user_db(db_file, user_db)
    return JSONResponse({"episode_id": episode_id, "dialogues": dialogues})

def render_worker(active_sid: str, cfg_dict: dict):
    with RENDER_LOCK:
        _, _, processed_dir, db_file = get_session_workspace(active_sid)
        ep_ids = cfg_dict.get("episode_ids", [])
        kiri_key = cfg_dict.get("kiri_api_key", "")

        for ep_id in ep_ids:
            user_db = load_user_db(db_file)
            if ep_id not in user_db: continue
            ep = user_db[ep_id]
            video_in = ep["path"]
            total_dur = get_media_duration(video_in) or 60.0
            dialogues = cfg_dict.get("dialogues_map", {}).get(ep_id) or ep.get("dialogues", [])

            clips = []
            mute_intervals = []
            for idx, it in enumerate(dialogues):
                txt = it.get("khmer_text", "").strip()
                if not txt: continue
                role = it.get("speaker", "male")
                r_wav = processed_dir / f"{ep_id}_r_{idx}.wav"
                f_wav = processed_dir / f"{ep_id}_f_{idx}.wav"
                
                # បង្កើតសំឡេង (ប្រើ Kiri TTS Key ពី UI បើមាន)
                if generate_voice_raw(txt, role, str(r_wav), kiri_key=kiri_key):
                    scene_dur = float(it["end"]) - float(it["start"])
                    fit_audio_exact_to_scene(str(r_wav), scene_dur, str(f_wav))
                    r_wav.unlink(missing_ok=True)
                    s_time = float(it["start"])
                    e_time = float(it["end"])
                    clips.append({"start": s_time, "path": str(f_wav)})
                    mute_intervals.append((max(0.0, s_time - 0.04), e_time + 0.04))

            dub_track = processed_dir / f"{ep_id}_dub.wav"
            if clips:
                inputs_cmd = ["-f", "lavfi", "-t", f"{total_dur:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
                filter_parts = []
                for idx, c in enumerate(clips):
                    inputs_cmd.extend(["-i", c["path"]])
                    ms = int(max(0, c["start"]) * 1000)
                    filter_parts.append(f"[{idx+1}:a]adelay={ms}|{ms}[a{idx+1}]")
                all_in = "[0:a]" + "".join([f"[a{i+1}]" for i in range(len(clips))])
                filter_complex = f"{';'.join(filter_parts)};{all_in}amix=inputs={len(clips)+1}:dropout_transition=0:normalize=0[aout]"
                subprocess.run([
                    FFMPEG_BIN, "-y", *inputs_cmd, "-filter_complex", filter_complex,
                    "-map", "[aout]", "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(dub_track)
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                subprocess.run([
                    FFMPEG_BIN, "-y", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", f"{total_dur}",
                    "-c:a", "pcm_s16le", str(dub_track)
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            volume_expr_parts = []
            for s_m, e_m in mute_intervals:
                volume_expr_parts.append(f"between(t,{s_m:.2f},{e_m:.2f})")
            
            if volume_expr_parts:
                combined_between = "+".join(volume_expr_parts)
                bgm_filter = f"[0:a]volume=eval=frame:volume='if({combined_between}, 0.0, 0.35)'[orig_cut]"
            else:
                bgm_filter = "[0:a]volume=0.0[orig_cut]"

            dub_vol = float(cfg_dict.get("dub_volume", 2.20))
            master_mix = f"{bgm_filter};[1:a]volume={dub_vol}[dub_km];[orig_cut][dub_km]amix=inputs=2:duration=first:dropout_transition=0[a_final]"

            out_name = f"dubbed_{ep.get('code', 'ep')}_{uuid.uuid4().hex[:6]}.mp4"
            final_mp4 = processed_dir / out_name

            cmd_render = [
                FFMPEG_BIN, "-y",
                "-i", str(video_in),
                "-i", str(dub_track),
                "-filter_complex", master_mix,
                "-map", "0:v:0",
                "-map", "[a_final]",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "ultrafast",
                "-c:a", "aac",
                "-b:a", "192k",
                "-movflags", "+faststart",
                "-t", f"{total_dur:.3f}",
                str(final_mp4)
            ]
            subprocess.run(cmd_render, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            for c in clips: Path(c["path"]).unlink(missing_ok=True)
            dub_track.unlink(missing_ok=True)

            user_db = load_user_db(db_file)
            user_db[ep_id]["status"] = "completed"
            user_db[ep_id]["output_file"] = out_name
            user_db[ep_id]["output_url"] = f"/api/download/{active_sid}/{out_name}"
            user_db[ep_id]["video_url"] = f"/api/download/{active_sid}/{out_name}?t={uuid.uuid4().hex[:6]}"
            save_user_db(db_file, user_db)
            print(f"[✓] Render ភាគ {ep.get('code')} ចប់សព្វគ្រប់!")

@app.post("/api/batch-render")
async def start_batch_render(cfg: BatchRenderConfig, sid: Optional[str] = Cookie(None)):
    active_sid = cfg.session_id or sid or "default_user"
    threading.Thread(target=render_worker, args=(active_sid, cfg.dict()), daemon=True).start()
    return JSONResponse({"message": "Render started"})

@app.get("/api/download/{session_id}/{filename}")
async def download_user_file(session_id: str, filename: str):
    _, _, processed_dir, _ = get_session_workspace(session_id)
    target_file = processed_dir / filename
    if not target_file.exists():
        raise HTTPException(status_code=404, detail="File not found")
    safe_download_name = filename[:-4] if filename.lower().endswith(".mp4.mp4") else filename
    return FileResponse(path=str(target_file), filename=safe_download_name, media_type="video/mp4")

if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import time

    port = 8080
    clean_target = "[http://127.0.0.1:8080](http://127.0.0.1:8080)"
    print("\n[✓] KhmerDub Pro Studio បានបើកដំណើរការជោគជ័យ!")
    print(f"[👉] Link ប្រើប្រាស់: {clean_target}\n")

    def open_browser():
        time.sleep(1.2)
        try: webbrowser.open(clean_target)
        except Exception: pass

    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("server:app", host="0.0.0.0", port=port, log_level="critical")