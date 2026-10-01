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
import logging
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

# --- កម្ចាត់ WinError 10054 និង ProactorBasePipeTransport Crash លើ Windows ---
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
                    try:
                        self._sock.shutdown(1)
                    except Exception:
                        pass
                    self._sock.close()
            except Exception:
                pass

        _ProactorBasePipeTransport._call_connection_lost = silence_connection_lost
    except Exception:
        pass
# -------------------------------------------------------------------------

FFMPEG_BIN = "ffmpeg"
try:
    import imageio_ffmpeg
    f_path = imageio_ffmpeg.get_ffmpeg_exe()
    if os.path.exists(f_path):
        try:
            os.chmod(f_path, 0o755)
        except Exception:
            pass
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

app = FastAPI(title="KhmerDub Studio Pro - Audible Voice & Pure BGM Engine", version="2700.0.0")

# Middleware ទប់ស្កាត់ Connection Reset Noise
@app.middleware("http")
async def suppress_connection_reset_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except (ConnectionResetError, BrokenPipeError):
        return Response(status_code=204)
    except Exception as e:
        if "10054" in str(e) or "connection_lost" in str(e):
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

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

RENDER_LOCK = threading.Lock()
WHISPER_LOCK = threading.Lock()
GLOBAL_WHISPER_INSTANCE = None

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

class BatchRenderConfig(BaseModel):
    session_id: Optional[str] = "default_user"
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

def resolve_khmer_voice(speaker_raw: str) -> str:
    s = str(speaker_raw).lower()
    if any(k in s for k in ["sreymom", "ស្រី", "female", "girl", "woman"]):
        return "km-KH-SreymomNeural"
    return "km-KH-PisethNeural"

def detect_voice_pitch_gender(audio_path: Path, start_sec: float, end_sec: float) -> str:
    dur = max(0.4, end_sec - start_sec)
    try:
        cut_cmd = [
            FFMPEG_BIN, "-y",
            "-ss", f"{start_sec:.3f}",
            "-t", f"{dur:.3f}",
            "-i", str(audio_path),
            "-af", "highpass=f=80,lowpass=f=400,astats=metadata=1:reset=1",
            "-f", "null", "-"
        ]
        res = subprocess.run(cut_cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True, errors="replace")
        zcr_matches = re.findall(r"Zero crossing rate:\s*(\d+\.?\d*)", res.stderr)
        if zcr_matches:
            avg_zcr = sum([float(x) for x in zcr_matches]) / len(zcr_matches)
            if avg_zcr >= 0.045:
                return "sreymom"
            elif avg_zcr < 0.040:
                return "piseth"

        spec_cmd = [
            FFMPEG_BIN, "-y",
            "-ss", f"{start_sec:.3f}",
            "-t", f"{dur:.3f}",
            "-i", str(audio_path),
            "-af", "bandpass=frequency=240:width_type=h:width=120,volumedetect",
            "-f", "null", "-"
        ]
        res_spec = subprocess.run(spec_cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True, errors="replace")
        mean_vol = re.search(r"mean_volume:\s*(-?\d+\.?\d*)\s*dB", res_spec.stderr)
        if mean_vol:
            db_val = float(mean_vol.group(1))
            if db_val > -36.0:
                return "sreymom"
            else:
                return "piseth"
    except Exception:
        pass
    return "piseth"

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
                success = future.result(timeout=4.0)
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

    # Convert ទៅជា PCM 16-bit 48000Hz Stereo ស្តង់ដារ
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

def get_shared_whisper():
    global GLOBAL_WHISPER_INSTANCE
    if GLOBAL_WHISPER_INSTANCE is None:
        if WhisperModel:
            try:
                GLOBAL_WHISPER_INSTANCE = WhisperModel("tiny", device="cpu", compute_type="int8", cpu_threads=4, num_workers=1)
            except Exception as e:
                print(f"[!] WhisperModel load error: {e}")
                GLOBAL_WHISPER_INSTANCE = None
    return GLOBAL_WHISPER_INSTANCE

def translate_dialogues_mwapi(texts: List[str], cloud_key: str, endpoint_base: str = "https://api.mwapi.dev", source_lang: str = "auto") -> List[str]:
    if not texts:
        return []

    lang_desc = "Chinese or English" if source_lang in ["auto", ""] else ("Chinese" if source_lang == "zh" else "English")
    system_prompt = (
        f"You are a professional film dubbing translator. Translate the following array of {lang_desc} "
        "dialogues into natural, spoken Khmer for direct character dubbing. "
        "Keep the phrases concise for sync. Return ONLY a strict JSON array of translated strings."
    )

    if cloud_key:
        for model_choice in ["claude-3-haiku-20240307", "gpt-4o-mini", "claude-3-5-sonnet-20240620"]:
            try:
                req_data = {
                    "model": model_choice,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": json.dumps(texts, ensure_ascii=False)}
                    ],
                    "temperature": 0.2
                }
                url = f"{endpoint_base.rstrip('/')}/v1/chat/completions"
                req = urllib.request.Request(
                    url,
                    data=json.dumps(req_data).encode("utf-8"),
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {cloud_key.strip()}"}
                )
                with urllib.request.urlopen(req, timeout=16) as resp:
                    res_json = json.loads(resp.read().decode("utf-8"))
                    content_str = res_json["choices"][0]["message"]["content"].strip()
                    match = re.search(r'\[.*\]', content_str, re.DOTALL)
                    if match:
                        parsed = json.loads(match.group(0))
                        if len(parsed) == len(texts):
                            return [str(p).strip() for p in parsed]
            except Exception:
                continue

    results = []
    src_code = "auto" if source_lang == "auto" else ("zh-CN" if source_lang == "zh" else "en")
    for txt in texts:
        translated = ""
        try:
            encoded = urllib.parse.quote(txt)
            gurl = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={src_code}&tl=km&dt=t&q={encoded}"
            req = urllib.request.Request(gurl, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data and data[0]:
                    translated = "".join([s[0] for s in data[0] if s and s[0]]).strip()
        except Exception:
            translated = txt
        results.append(translated or txt)
    return results

def extract_pure_bgm_sfx_fast(video_path: str, output_bgm_wav: Path):
    vocal_strip_filter = (
        "stereotools=mlev=0.01:slev=1.20,"
        "highpass=f=45,lowpass=f=17500,"
        "equalizer=f=350:t=q:w=1.8:g=-14,"
        "equalizer=f=1100:t=q:w=1.5:g=-15,"
        "volume=0.9"
    )
    subprocess.run([
        FFMPEG_BIN, "-y", "-i", str(video_path),
        "-vn",
        "-af", vocal_strip_filter,
        "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le",
        str(output_bgm_wav)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_bgm_wav.exists() and output_bgm_wav.stat().st_size > 1000

@app.get("/")
async def root_view(response: Response, sid: Optional[str] = Cookie(None)):
    session_id = sid or str(uuid.uuid4().hex[:12])
    response.set_cookie(key="sid", value=session_id, max_age=86400 * 30, httponly=False)
    index_path = STATIC_DIR / "index.html"
    return FileResponse(index_path)

@app.post("/api/batch-upload")
async def batch_upload(
    files: List[UploadFile] = File(...),
    session_id: str = Form(""),
    sid: Optional[str] = Cookie(None)
):
    active_sid = session_id or sid or "default_user"
    _, uploads_dir, _, db_file = get_session_workspace(active_sid)
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
            "video_url": f"/api/video/{active_sid}/{ep_id}",
            "status": "ready",
            "progress": 0,
            "dialogues": []
        }
        uploaded_eps.append(user_db[ep_id])

    save_user_db(db_file, user_db)
    return {"uploaded": uploaded_eps}

@app.get("/api/video/{session_id}/{ep_id}")
async def stream_video(session_id: str, ep_id: str):
    _, _, _, db_file = get_session_workspace(session_id)
    user_db = load_user_db(db_file)
    if ep_id not in user_db or not os.path.exists(user_db[ep_id]["path"]):
        raise HTTPException(status_code=404, detail="Video not found")

    video_path = Path(user_db[ep_id]["path"])
    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        headers={
            "Accept-Ranges": "bytes",
            "Connection": "keep-alive"
        }
    )

@app.get("/api/episodes")
async def get_episodes(session_id: str = "", sid: Optional[str] = Cookie(None)):
    active_sid = session_id or sid or "default_user"
    _, _, _, db_file = get_session_workspace(active_sid)
    return JSONResponse(load_user_db(db_file))

@app.post("/api/clear-all-episodes")
async def clear_all_episodes(session_id: str = Form(""), sid: Optional[str] = Cookie(None)):
    active_sid = session_id or sid or "default_user"
    user_dir, _, _, _ = get_session_workspace(active_sid)
    try:
        shutil.rmtree(user_dir)
    except Exception:
        pass
    return {"message": "All episodes cleared successfully"}

@app.delete("/api/episodes/{episode_id}")
async def delete_single_episode_api(episode_id: str, session_id: str = "", sid: Optional[str] = Cookie(None)):
    active_sid = session_id or sid or "default_user"
    _, _, _, db_file = get_session_workspace(active_sid)
    user_db = load_user_db(db_file)
    if episode_id in user_db:
        ep = user_db.pop(episode_id)
        try:
            if os.path.exists(ep.get("path", "")):
                os.remove(ep["path"])
        except Exception:
            pass
        new_db = {}
        for idx, (k, v) in enumerate(user_db.items()):
            v["code"] = f"EP{idx + 1:02d}"
            new_db[k] = v
        save_user_db(db_file, new_db)
        return {"message": "Deleted successfully", "episodes": new_db}
    raise HTTPException(status_code=404, detail="Episode not found")

@app.post("/api/transcribe-episode")
async def transcribe_episode(
    episode_id: str = Form(...),
    session_id: str = Form(""),
    model_size: str = Form("mwapi"),
    cloud_key: str = Form(""),
    api_endpoint: str = Form("https://api.mwapi.dev"),
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
    ep["status"] = "transcribing"
    save_user_db(db_file, user_db)

    video_path = ep["path"]
    audio_path = uploads_dir / f"{episode_id}_audio.wav"

    subprocess.run([
        FFMPEG_BIN, "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(audio_path)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    sub_segments = []

    with WHISPER_LOCK:
        whisper = get_shared_whisper()
        if whisper:
            try:
                whisper_lang = None if lang == "auto" else lang
                segments, _ = whisper.transcribe(
                    str(audio_path),
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=250),
                    language=whisper_lang,
                    beam_size=2,
                    temperature=0.0
                )
                for seg in segments:
                    t = seg.text.strip()
                    if t:
                        sub_segments.append({
                            "id": len(sub_segments) + 1,
                            "start": round(seg.start, 2),
                            "end": round(seg.end, 2),
                            "text": t
                        })
            except Exception as e:
                print(f"[!] Whisper Error: {e}")

    if not sub_segments:
        total_dur = get_media_duration(str(audio_path)) or 10.0
        sub_segments.append({
            "id": 1,
            "start": 0.0,
            "end": round(min(3.5, total_dur), 2),
            "text": "你好" if lang in ["zh", "auto"] else "Hello"
        })

    orig_texts = [s["text"] for s in sub_segments]
    khmer_translations = translate_dialogues_mwapi(orig_texts, cloud_key=cloud_key, endpoint_base=api_endpoint, source_lang=lang)

    dialogues = []
    female_kw = r"(她|女人|女孩|小姐|妈妈|母亲|妻子|老婆|姐姐|妹妹|夫人|太太|阿姨|小姐|女|she|her|woman|girl|lady|miss|mrs|mother|sister|wife)"
    male_kw = r"(他|男人|男孩|先生|爸爸|父亲|丈夫|老公|哥哥|弟弟|少爷|老爹|林总|王总|徐总|董事长|总裁|男|he|him|his|man|boy|mr|sir|father|brother|husband)"

    for idx, orig in enumerate(sub_segments):
        f_km = khmer_translations[idx] if idx < len(khmer_translations) else orig["text"]

        if speaker_mode == "male":
            detected_speaker = "piseth"
        elif speaker_mode == "female":
            detected_speaker = "sreymom"
        else:
            pitch_gender = detect_voice_pitch_gender(audio_path, orig["start"], orig["end"])
            if re.search(female_kw, orig["text"].lower()):
                detected_speaker = "sreymom"
            elif re.search(male_kw, orig["text"].lower()):
                detected_speaker = "piseth"
            else:
                detected_speaker = pitch_gender

        dialogues.append({
            "id": idx + 1,
            "start": orig["start"],
            "end": orig["end"],
            "original_text": orig["text"],
            "khmer_text": f_km,
            "speaker": detected_speaker
        })

    user_db = load_user_db(db_file)
    if episode_id in user_db:
        user_db[episode_id]["dialogues"] = dialogues
        user_db[episode_id]["status"] = "transcribed"
        save_user_db(db_file, user_db)

    return JSONResponse({"episode_id": episode_id, "dialogues": dialogues})

@app.post("/api/preview-tts")
async def preview_single_tts(
    text: str = Form(...),
    speaker: str = Form("piseth"),
    session_id: str = Form(""),
    sid: Optional[str] = Cookie(None)
):
    active_sid = session_id or sid or "default_user"
    _, _, processed_dir, _ = get_session_workspace(active_sid)
    clean_text = text.strip() or "សួស្តី"
    voice = resolve_khmer_voice(speaker)
    uid = uuid.uuid4().hex[:6]
    out_wav = processed_dir / f"prev_{uid}.wav"

    ok = generate_khmer_audio_fast(clean_text, voice, str(out_wav))
    if not ok or not out_wav.exists():
        raise HTTPException(status_code=500, detail="TTS Engine Failed")

    return FileResponse(path=str(out_wav), media_type="audio/wav", filename="preview.wav")

@app.get("/api/download/{session_id}/{filename}")
async def download_user_file(session_id: str, filename: str):
    _, _, processed_dir, _ = get_session_workspace(session_id)
    file_path = processed_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=str(file_path), filename=filename, media_type="video/mp4")

# Turbo Render: ធានាថាសំឡេងខ្មែរ (ពិសិដ្ឋ/ស្រីមុំ) ឮច្បាស់ពេញទំហឹង + រក្សា BGM & SFX
def render_worker(active_sid: str, cfg_dict: dict):
    with RENDER_LOCK:
        _, _, processed_dir, db_file = get_session_workspace(active_sid)
        ep_ids = cfg_dict.get("episode_ids", [])

        for ep_id in ep_ids:
            user_db = load_user_db(db_file)
            if ep_id not in user_db:
                continue

            ep = user_db[ep_id]
            ep["status"] = "rendering"
            ep["progress"] = 15
            save_user_db(db_file, user_db)

            dialogues = cfg_dict.get("dialogues_map", {}).get(ep_id) or ep.get("dialogues", [])
            video_in = ep["path"]
            total_video_duration = get_media_duration(str(video_in)) or 60.0

            # ១. ទាញយក BGM & SFX ដើម
            pure_bgm_track = processed_dir / f"{ep_id}_bgm_pure.wav"
            extract_pure_bgm_sfx_fast(video_in, pure_bgm_track)

            user_db = load_user_db(db_file)
            user_db[ep_id]["progress"] = 35
            save_user_db(db_file, user_db)

            # ២. បង្កើតសំឡេងខ្មែរស្របគ្នា ៨ ឃ្លាក្នុងពេលតែមួយ
            def process_single_clip(idx_item):
                idx, d = idx_item
                it = d if isinstance(d, dict) else d.dict()
                txt = it.get("khmer_text", "").strip()
                if not txt:
                    return None
                
                spk_val = it.get("speaker", "piseth")
                v_id = resolve_khmer_voice(spk_val)

                r_wav = processed_dir / f"{ep_id}_r_{idx}.wav"
                f_wav = processed_dir / f"{ep_id}_f_{idx}.wav"

                if generate_khmer_audio_fast(txt, v_id, str(r_wav)):
                    scene_dur = max(0.35, float(it["end"]) - float(it["start"]))
                    fit_audio_exact_to_scene(str(r_wav), scene_dur, str(f_wav))
                    try:
                        r_wav.unlink(missing_ok=True)
                    except Exception:
                        pass
                    return {
                        "start": float(it["start"]),
                        "duration": get_media_duration(str(f_wav)),
                        "path": str(f_wav)
                    }
                return None

            clips_results = []
            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(process_single_clip, enumerate(dialogues)))
                clips_results = [r for r in results if r is not None]

            clips_results.sort(key=lambda x: x["start"])

            user_db = load_user_db(db_file)
            user_db[ep_id]["progress"] = 70
            save_user_db(db_file, user_db)

            # ៣. ផ្គុំ Dub Track ខ្មែរ ដោយបញ្ចូល Base Silence មួយដើម្បីធានាថា Track មិនដាច់
            dub_track = processed_dir / f"{ep_id}_dub.wav"
            if clips_results:
                inputs_cmd = [
                    "-f", "lavfi",
                    "-t", f"{total_video_duration:.3f}",
                    "-i", "anullsrc=r=48000:cl=stereo"
                ]
                filter_parts = []
                for idx, c in enumerate(clips_results):
                    inputs_cmd.extend(["-i", c["path"]])
                    delay_ms = int(max(0, c["start"]) * 1000)
                    filter_parts.append(f"[{idx+1}:a]adelay={delay_ms}|{delay_ms}[a{idx+1}]")
                
                all_inputs = "[0:a]" + "".join([f"[a{i+1}]" for i in range(len(clips_results))])
                filter_complex = f"{';'.join(filter_parts)};{all_inputs}amix=inputs={len(clips_results)+1}:dropout_transition=0:normalize=0[aout]"
                
                cmd_mix = [
                    FFMPEG_BIN, "-y",
                    *inputs_cmd,
                    "-filter_complex", filter_complex,
                    "-map", "[aout]",
                    "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le",
                    str(dub_track)
                ]
                subprocess.run(cmd_mix, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                subprocess.run([
                    FFMPEG_BIN, "-y", "-f", "lavfi",
                    "-i", f"anullsrc=r=48000:cl=stereo", "-t", f"{total_video_duration}",
                    "-c:a", "pcm_s16le", str(dub_track)
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            user_db = load_user_db(db_file)
            user_db[ep_id]["progress"] = 85
            save_user_db(db_file, user_db)

            out_name = f"dubbed_{ep.get('code', 'ep')}_{uuid.uuid4().hex[:6]}.mp4"
            final_mp4 = processed_dir / out_name

            dub_vol = float(cfg_dict.get("dub_volume", 2.20))

            # ៤. Master Audio Mix: ធានាថាសំឡេងខ្មែរ [1:a] ឮខ្លាំងច្បាស់ 100% លើកដំបូង + BGM [0:a] បន្ថយមកត្រឹម 0.28
            master_filter = (
                f"[1:a]volume={dub_vol}[dub_loud];"
                f"[0:a]volume=0.28[bgm_soft];"
                f"[bgm_soft][dub_loud]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,alimiter=limit=0.98[final_a]"
            )

            cmd_final = [
                FFMPEG_BIN, "-y",
                "-i", str(pure_bgm_track), # 0: ភ្លេង BGM + SFX ដើម
                "-i", str(dub_track),      # 1: សំឡេងខ្មែរ (ពិសិដ្ឋ/ស្រីមុំ)
                "-i", str(video_in),       # 2: វីដេអូរូបភាពដើម
                "-map", "2:v:0",
                "-filter_complex", master_filter,
                "-map", "[final_a]",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "256k",
                "-shortest",
                str(final_mp4)
            ]
            res = subprocess.run(cmd_final, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)

            # Fallback ក្នុងករណី filter_complex មានបញ្ហា
            if not final_mp4.exists() or final_mp4.stat().st_size < 1000:
                cmd_fallback = [
                    FFMPEG_BIN, "-y",
                    "-i", str(video_in),
                    "-i", str(dub_track),
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    "-filter:a", f"volume={dub_vol},alimiter=limit=0.98",
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-shortest",
                    str(final_mp4)
                ]
                subprocess.run(cmd_fallback, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # សម្អាត Files
            for c in clips_results:
                try:
                    Path(c["path"]).unlink(missing_ok=True)
                except Exception:
                    pass
            try:
                pure_bgm_track.unlink(missing_ok=True)
                dub_track.unlink(missing_ok=True)
            except Exception:
                pass

            user_db = load_user_db(db_file)
            if ep_id in user_db:
                user_db[ep_id]["status"] = "completed"
                user_db[ep_id]["progress"] = 100
                user_db[ep_id]["output_url"] = f"/api/download/{active_sid}/{out_name}"
                save_user_db(db_file, user_db)

@app.post("/api/batch-render")
async def start_batch_render(cfg: BatchRenderConfig, sid: Optional[str] = Cookie(None)):
    active_sid = cfg.session_id or sid or "default_user"
    worker = threading.Thread(target=render_worker, args=(active_sid, cfg.dict()), daemon=True)
    worker.start()
    return JSONResponse({"message": "Render started"})

if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import time

    logging.getLogger("asyncio").setLevel(logging.CRITICAL)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

    port = int(os.environ.get("PORT", 8080))

    if sys.platform == "win32":
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception:
            pass

    def open_browser():
        time.sleep(1.2)
        target_url = f"http://127.0.0.1:{port}"
        print(f"\n[✓] KhmerDub Studio Pro កំពុងដំណើរការ: {target_url}\n")
        try:
            webbrowser.open(target_url)
        except Exception:
            pass

    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=port,
        loop="asyncio",
        log_level="warning",
        reload=False
    )