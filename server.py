import os
import sys
import uuid
import shutil
import asyncio
import subprocess
import threading
import urllib.parse
import urllib.request
import json
import re
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

try:
    import imageio_ffmpeg
    FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_BIN = "ffmpeg"

if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Cookie, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

try:
    import edge_tts
except ImportError:
    edge_tts = None

app = FastAPI(title="KhmerDub Studio Pro - Multi-User Cloud Architecture", version="400.0.0")

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

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Global Processing Locks ដើម្បីការពារកុំឱ្យ CPU/RAM គាំងពេលមនុស្សច្រើនចុចព្រមគ្នា
RENDER_LOCK = threading.Lock()
WHISPER_LOCK = threading.Lock()

GLOBAL_WHISPER_INSTANCE = None
CURRENT_MODEL_NAME = None

def get_user_workspace(session_id: str):
    user_dir = WORKSPACE_ROOT / session_id
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
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
    return {}

def save_user_db(db_file: Path, data: dict):
    try:
        with open(db_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[!] Error saving DB: {e}")

class BatchRenderConfig(BaseModel):
    episode_ids: List[str]
    merge_all_into_one: bool = False
    speaker_mode: str = "auto"
    speed_factor: float = 1.0
    dub_volume: float = 2.20
    burn_subtitles: bool = True
    sub_color: str = "&H00FFFF"
    aspect_ratio: str = "9:16"
    resolution: str = "1080p"
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

def _run_edge_tts_isolated(text: str, voice: str, target_mp3: str) -> bool:
    async def _async_call():
        communicate = edge_tts.Communicate(text, voice, rate="+14%", pitch="+0Hz")
        await communicate.save(target_mp3)

    try:
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        new_loop.run_until_complete(_async_call())
        new_loop.close()
        return os.path.exists(target_mp3) and os.path.getsize(target_mp3) > 150
    except Exception:
        return False

def generate_khmer_audio_fast(text: str, voice: str, out_wav_path: str) -> bool:
    clean_text = text.strip() or "បាទ"
    temp_mp3 = Path(out_wav_path).with_suffix(".mp3")
    success = False

    if edge_tts:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_edge_tts_isolated, clean_text, voice, str(temp_mp3))
            try:
                success = future.result(timeout=3.5)
            except Exception:
                success = False

    if not success or not os.path.exists(temp_mp3) or os.path.getsize(temp_mp3) < 150:
        try:
            encoded = urllib.parse.quote(clean_text[:180])
            url = f"https://translate.google.com/translate_tts?ie=UTF-8&q={encoded}&tl=km&client=tw-ob"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=3.0) as resp, open(str(temp_mp3), "wb") as f:
                f.write(resp.read())
            if os.path.exists(temp_mp3) and os.path.getsize(temp_mp3) > 150:
                success = True
        except Exception:
            pass

    if not success or not os.path.exists(temp_mp3):
        subprocess.run([
            FFMPEG_BIN, "-y", "-f", "lavfi",
            "-i", "anullsrc=r=48000:cl=stereo",
            "-t", "0.5",
            "-c:a", "pcm_s16le",
            str(out_wav_path)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True

    subprocess.run([
        FFMPEG_BIN, "-y", "-i", str(temp_mp3),
        "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le",
        str(out_wav_path)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    try:
        temp_mp3.unlink(missing_ok=True)
    except Exception:
        pass
    return os.path.exists(out_wav_path) and os.path.getsize(out_wav_path) > 100

def fit_audio_exact_to_scene(raw_wav: str, scene_dur: float, out_wav: str):
    actual_dur = get_media_duration(raw_wav)
    if actual_dur <= 0.05 or scene_dur <= 0.05:
        shutil.copyfile(raw_wav, out_wav)
        return

    target_dur = max(0.25, scene_dur - 0.04)
    if actual_dur > scene_dur:
        ratio = min(actual_dur / target_dur, 2.2)
        filters = []
        curr = ratio
        while curr > 2.0:
            filters.append("atempo=2.0")
            curr /= 2.0
        while curr < 0.5:
            filters.append("atempo=0.5")
            curr /= 0.5
        filters.append(f"atempo={curr:.3f}")
        af_str = ",".join(filters)
    else:
        af_str = "anull"

    cmd = [
        FFMPEG_BIN, "-y", "-i", str(raw_wav),
        "-filter:a", af_str,
        "-t", f"{scene_dur:.3f}",
        "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le",
        str(out_wav)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.exists(out_wav) or os.path.getsize(out_wav) < 100:
        shutil.copyfile(raw_wav, out_wav)

def get_shared_whisper_instance():
    global GLOBAL_WHISPER_INSTANCE
    if GLOBAL_WHISPER_INSTANCE is None:
        if WhisperModel:
            try:
                GLOBAL_WHISPER_INSTANCE = WhisperModel("tiny", device="cpu", compute_type="int8", cpu_threads=2, num_workers=1)
            except Exception:
                pass
    return GLOBAL_WHISPER_INSTANCE

def translate_batch_with_ai_cloud(texts: List[str], source_lang: str = "auto", cloud_type: str = "", cloud_key: str = "") -> List[str]:
    if not texts:
        return []

    lang_desc = "Chinese or English" if source_lang in ["auto", ""] else ("Chinese" if source_lang == "zh" else "English")
    system_prompt = (
        f"You are a professional film voiceover translator. Translate the following array of {lang_desc} "
        "movie dialogue lines into natural spoken Khmer for direct character dubbing. "
        "RULES:\n"
        "1. Match conversational emotion and context.\n"
        "2. Keep Khmer translations concise and punchy to fit lip-sync timing.\n"
        "3. Output MUST be ONLY a strict JSON array of translated Khmer strings."
    )

    if cloud_type == "cloud_openai" and cloud_key:
        try:
            req_data = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(texts, ensure_ascii=False)}
                ],
                "temperature": 0.2
            }
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=json.dumps(req_data).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {cloud_key.strip()}"}
            )
            with urllib.request.urlopen(req, timeout=12) as resp:
                res_json = json.loads(resp.read().decode("utf-8"))
                match = re.search(r'\[.*\]', res_json["choices"][0]["message"]["content"].strip(), re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                    if len(parsed) == len(texts):
                        return [str(p) for p in parsed]
        except Exception:
            pass

    if cloud_type == "cloud_groq" and cloud_key:
        try:
            req_data = {
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(texts, ensure_ascii=False)}
                ],
                "temperature": 0.2
            }
            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/chat/completions",
                data=json.dumps(req_data).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {cloud_key.strip()}"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                res_json = json.loads(resp.read().decode("utf-8"))
                match = re.search(r'\[.*\]', res_json["choices"][0]["message"]["content"].strip(), re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                    if len(parsed) == len(texts):
                        return [str(p) for p in parsed]
        except Exception:
            pass

    results = []
    src_code = "auto" if source_lang == "auto" else ("zh-CN" if source_lang == "zh" else "en")
    for txt in texts:
        clean = re.sub(r'^[（\(].*?[）\)]', '', txt.strip())
        if not clean:
            results.append("")
            continue
        translated = ""
        try:
            encoded = urllib.parse.quote(clean)
            url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={src_code}&tl=km&dt=t&q={encoded}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data and data[0]:
                    translated = "".join([s[0] for s in data[0] if s and s[0]]).strip()
        except Exception:
            translated = clean
        results.append(translated or clean)
    return results

@app.get("/")
async def root_view(response: Response, sid: Optional[str] = Cookie(None)):
    session_id = sid or str(uuid.uuid4().hex[:12])
    response.set_cookie(key="sid", value=session_id, max_age=86400 * 7, httponly=False)
    index_path = STATIC_DIR / "index.html"
    return FileResponse(index_path)

@app.post("/api/batch-upload")
async def batch_upload(files: List[UploadFile] = File(...), sid: Optional[str] = Cookie(None)):
    session_id = sid or "default_user"
    _, uploads_dir, _, db_file = get_user_workspace(session_id)
    user_db = load_user_db(db_file)

    uploaded_eps = []
    current_count = len(user_db)
    for i, f in enumerate(files):
        ep_id = f"ep_{uuid.uuid4().hex[:6]}"
        ep_code = f"EP{current_count + i + 1:02d}"
        ext = Path(f.filename).suffix.lower() or ".mp4"
        dest_path = uploads_dir / f"{ep_id}_{ep_code}{ext}"

        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(f.file, buffer)

        user_db[ep_id] = {
            "id": ep_id,
            "code": ep_code,
            "filename": f.filename,
            "path": str(dest_path),
            "status": "ready",
            "progress": 0,
            "dialogues": []
        }
        uploaded_eps.append(user_db[ep_id])

    save_user_db(db_file, user_db)
    return {"uploaded": uploaded_eps}

@app.get("/api/episodes")
async def get_episodes(sid: Optional[str] = Cookie(None)):
    session_id = sid or "default_user"
    _, _, _, db_file = get_user_workspace(session_id)
    return JSONResponse(load_user_db(db_file))

@app.post("/api/clear-all-episodes")
async def clear_all_episodes(sid: Optional[str] = Cookie(None)):
    session_id = sid or "default_user"
    user_dir, _, _, _ = get_user_workspace(session_id)
    try:
        shutil.rmtree(user_dir)
    except Exception:
        pass
    return {"message": "Cleared successfully"}

@app.post("/api/transcribe-episode")
async def transcribe_episode(
    episode_id: str = Form(...),
    model_size: str = Form("tiny"),
    cloud_key: str = Form(""),
    lang: str = Form("auto"),
    speaker_mode: str = Form("auto"),
    sid: Optional[str] = Cookie(None)
):
    session_id = sid or "default_user"
    _, uploads_dir, _, db_file = get_user_workspace(session_id)
    user_db = load_user_db(db_file)

    if episode_id not in user_db:
        raise HTTPException(status_code=404, detail="Episode not found")

    ep = user_db[episode_id]
    ep["status"] = "transcribing"
    save_user_db(db_file, user_db)

    video_path = ep["path"]
    audio_path = uploads_dir / f"{episode_id}_audio.wav"

    subprocess.run([
        FFMPEG_BIN, "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(audio_path)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # ប្រើ Lock ដើម្បីឱ្យអ្នកប្រើផ្សេងទៀតតម្រង់ជួរគ្នា មិននាំឱ្យលើសទំហំ RAM
    with WHISPER_LOCK:
        whisper = get_shared_whisper_instance()
        sub_segments = []
        if whisper:
            whisper_lang = None if lang == "auto" else lang
            segments, _ = whisper.transcribe(str(audio_path), vad_filter=True, language=whisper_lang, beam_size=1, temperature=0.0)
            for seg in segments:
                t = seg.text.strip()
                if t:
                    sub_segments.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "text": t})

    orig_texts = [s["text"] for s in sub_segments]
    khmer_translations = translate_batch_with_ai_cloud(orig_texts, source_lang=lang, cloud_type=model_size, cloud_key=cloud_key)

    dialogues = []
    last_spk = "piseth"
    for idx, orig in enumerate(sub_segments):
        f_km = khmer_translations[idx] if idx < len(khmer_translations) else orig["text"]
        spk = "piseth" if speaker_mode == "male" else ("sreymom" if speaker_mode == "female" else ("sreymom" if last_spk == "piseth" else "piseth"))
        last_spk = spk
        dialogues.append({
            "id": idx + 1,
            "start": orig["start"],
            "end": orig["end"],
            "original_text": orig["text"],
            "khmer_text": f_km,
            "speaker": spk
        })

    user_db = load_user_db(db_file)
    if episode_id in user_db:
        user_db[episode_id]["dialogues"] = dialogues
        user_db[episode_id]["status"] = "transcribed"
        save_user_db(db_file, user_db)

    return JSONResponse({"episode_id": episode_id, "dialogues": dialogues})

@app.get("/api/download/{session_id}/{filename}")
async def download_user_file(session_id: str, filename: str):
    _, _, processed_dir, _ = get_user_workspace(session_id)
    file_path = processed_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=str(file_path), filename=filename, media_type="video/mp4")

def render_worker(session_id: str, cfg_dict: dict):
    with RENDER_LOCK:
        _, _, processed_dir, db_file = get_user_workspace(session_id)
        ep_ids = cfg_dict.get("episode_ids", [])
        
        for ep_id in ep_ids:
            user_db = load_user_db(db_file)
            if ep_id not in user_db:
                continue

            ep = user_db[ep_id]
            ep["status"] = "rendering"
            ep["progress"] = 30
            save_user_db(db_file, user_db)

            dialogues = cfg_dict.get("dialogues_map", {}).get(ep_id) or ep.get("dialogues", [])
            video_in = ep["path"]
            total_dur = get_media_duration(video_in) or 60.0

            clips = []
            for idx, d in enumerate(dialogues):
                txt = d.get("khmer_text", "").strip()
                if not txt:
                    continue
                v_id = "km-KH-SreymomNeural" if d.get("speaker") == "sreymom" else "km-KH-PisethNeural"
                r_wav = processed_dir / f"{ep_id}_r_{idx}.wav"
                f_wav = processed_dir / f"{ep_id}_f_{idx}.wav"

                if generate_khmer_audio_fast(txt, v_id, str(r_wav)):
                    fit_audio_exact_to_scene(str(r_wav), max(0.35, float(d["end"]) - float(d["start"])), str(f_wav))
                    clips.append({"start": float(d["start"]), "duration": get_media_duration(str(f_wav)), "path": str(f_wav)})

            user_db["progress"] = 70
            save_user_db(db_file, user_db)

            dub_track = processed_dir / f"{ep_id}_dub.wav"
            manifest = processed_dir / f"{ep_id}_mf.txt"
            cursor = 0.0

            sil_master = processed_dir / "sil.wav"
            if not sil_master.exists():
                subprocess.run([FFMPEG_BIN, "-y", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", "30", "-c:a", "pcm_s16le", str(sil_master)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            with open(manifest, "w", encoding="utf-8") as f_mf:
                for idx, c in enumerate(clips):
                    gap = c["start"] - cursor
                    if gap > 0.04:
                        s_chunk = processed_dir / f"{ep_id}_s_{idx}.wav"
                        subprocess.run([FFMPEG_BIN, "-y", "-i", str(sil_master), "-t", f"{gap:.3f}", "-c", "copy", str(s_chunk)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        f_mf.write(f"file '{str(s_chunk).replace(os.sep, '/')}'\n")
                    f_mf.write(f"file '{c['path'].replace(os.sep, '/')}'\n")
                    cursor = c["start"] + c["duration"]

                if total_dur > cursor:
                    e_chunk = processed_dir / f"{ep_id}_e.wav"
                    subprocess.run([FFMPEG_BIN, "-y", "-i", str(sil_master), "-t", f"{total_dur - cursor:.3f}", "-c", "copy", str(e_chunk)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    f_mf.write(f"file '{str(e_chunk).replace(os.sep, '/')}'\n")

            subprocess.run([FFMPEG_BIN, "-y", "-f", "concat", "-safe", "0", "-i", str(manifest), "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2", str(dub_track)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            out_name = f"dubbed_{ep.get('code', 'ep')}_{uuid.uuid4().hex[:6]}.mp4"
            final_mp4 = processed_dir / out_name

            subprocess.run([
                FFMPEG_BIN, "-y",
                "-i", str(video_in),
                "-i", str(dub_track),
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-filter:a", "volume=2.2,alimiter=limit=0.98",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                str(final_mp4)
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            user_db = load_user_db(db_file)
            if ep_id in user_db:
                user_db[ep_id]["status"] = "completed"
                user_db[ep_id]["progress"] = 100
                user_db[ep_id]["output_url"] = f"/api/download/{session_id}/{out_name}"
                save_user_db(db_file, user_db)

@app.post("/api/batch-render")
async def start_batch_render(cfg: BatchRenderConfig, sid: Optional[str] = Cookie(None)):
    session_id = sid or "default_user"
    worker = threading.Thread(target=render_worker, args=(session_id, cfg.dict()), daemon=True)
    worker.start()
    return JSONResponse({"message": "Render enqueued successfully"})

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)