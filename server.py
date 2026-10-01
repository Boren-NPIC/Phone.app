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
from concurrent.futures import ThreadPoolExecutor, as_completed

# រក FFmpeg executable
FFMPEG_BIN = "ffmpeg"
try:
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    if os.path.exists(ffmpeg_exe):
        try:
            os.chmod(ffmpeg_exe, 0o755)
        except Exception:
            pass
        FFMPEG_BIN = ffmpeg_exe
except Exception:
    pass

if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Cookie, Response, Header
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

app = FastAPI(title="KhmerDub Studio Pro - Full Engine", version="500.0.0")

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
    trim_in: float = 0.0
    trim_out: Optional[float] = None

def format_ass_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds - int(seconds)) * 100)
    return f"{h:01d}:{m:02d}:{s:02d}.{cs:02d}"

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

def get_shared_whisper():
    global GLOBAL_WHISPER_INSTANCE
    if GLOBAL_WHISPER_INSTANCE is None:
        if WhisperModel:
            try:
                GLOBAL_WHISPER_INSTANCE = WhisperModel("tiny", device="cpu", compute_type="int8", cpu_threads=2, num_workers=1)
            except Exception as e:
                print(f"[!] Whisper init failed: {e}")
    return GLOBAL_WHISPER_INSTANCE

def analyze_smart_speaker(orig_text: str, khmer_text: str, last_speaker: str) -> str:
    female_regex = r"(她|女人|女孩|小姐|妈妈|妻子|老婆|姐姐|妹妹|夫人|太太|娘|珊珊|小雪|she|her|woman|girl|lady|miss|mrs|mother|sister|wife)"
    male_regex = r"(他|男人|男孩|先生|爸爸|丈夫|老公|哥哥|弟弟|少爷|老爹|林总|王总|he|him|his|man|boy|mr|sir|father|brother|husband)"
    female_km = r"(នាង|នារី|ស្រី|ម៉ាក់|ម្ដាយ|បងស្រី|ប្អូនស្រី|ភរិយា|ប្រពន្ធ|កញ្ញា|អ្នកស្រី|អូន|ចៅស្រី)"
    male_km = r"(គាត់|បុរស|ប្រុស|ប៉ា|ឪពុក|បងប្រុស|ប្អូនប្រុស|ស្វាមី|ប្ដី|លោក|បង|ចៅប្រុស)"

    orig_lower = orig_text.lower()
    if re.search(female_regex, orig_lower) or re.search(female_km, khmer_text):
        return "sreymom"
    if re.search(male_regex, orig_lower) or re.search(male_km, khmer_text):
        return "piseth"

    return "sreymom" if last_speaker == "piseth" else "piseth"

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
        "3. Output ONLY a strict JSON array of translated Khmer strings."
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
async def serve_video(session_id: str, ep_id: str):
    _, _, _, db_file = get_session_workspace(session_id)
    user_db = load_user_db(db_file)
    if ep_id in user_db and os.path.exists(user_db[ep_id]["path"]):
        return FileResponse(user_db[ep_id]["path"], media_type="video/mp4")
    raise HTTPException(status_code=404, detail="Video not found")

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
    model_size: str = Form("tiny"),
    cloud_key: str = Form(""),
    lang: str = Form("auto"),
    speaker_mode: str = Form("auto"),
    sid: Optional[str] = Cookie(None)
):
    try:
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

        # Extra fail-safe command សម្រាប់ Render Linux
        cmd = [
            FFMPEG_BIN, "-y", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(audio_path)
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 100:
            raise RuntimeError(f"FFmpeg extraction failed: {res.stderr}")

        sub_segments = []
        with WHISPER_LOCK:
            whisper = get_shared_whisper()
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
            spk = "piseth" if speaker_mode == "male" else ("sreymom" if speaker_mode == "female" else analyze_smart_speaker(orig["text"], f_km, last_spk))
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
    except Exception as e:
        print(f"[!] Transcribe Error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

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
    voice = "km-KH-SreymomNeural" if speaker == "sreymom" else "km-KH-PisethNeural"
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

def render_worker(active_sid: str, cfg_dict: dict):
    with RENDER_LOCK:
        _, _, processed_dir, db_file = get_session_workspace(active_sid)
        ep_ids = cfg_dict.get("episode_ids", [])
        completed_files = []

        for ep_id in ep_ids:
            user_db = load_user_db(db_file)
            if ep_id not in user_db:
                continue

            ep = user_db[ep_id]
            ep["status"] = "rendering"
            ep["progress"] = 25
            save_user_db(db_file, user_db)

            dialogues = cfg_dict.get("dialogues_map", {}).get(ep_id) or ep.get("dialogues", [])
            video_in = ep["path"]
            ass_file = processed_dir / f"{ep_id}.ass"

            res_w, res_h = (1080, 1920) if cfg_dict.get("aspect_ratio") != "16:9" else (1920, 1080)
            font_size = 50 if cfg_dict.get("resolution") != "4k" else 94

            with open(ass_file, "w", encoding="utf-8") as f:
                f.write(f"[Script Info]\nTitle: Direct Dub\nScriptType: v4.00+\nPlayResX: {res_w}\nPlayResY: {res_h}\n\n")
                f.write("[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, Italic, Alignment, MarginV, Outline, Shadow\n")
                f.write(f"Style: Default,Khmer OS Siemreap,{font_size},{cfg_dict.get('sub_color', '&H00FFFF')},&H00000000,&H80000000,1,0,2,85,2.5,1.2\n\n")
                f.write("[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
                for d in dialogues:
                    it = d if isinstance(d, dict) else d.dict()
                    s = format_ass_time(it["start"])
                    e = format_ass_time(it["end"])
                    txt = it.get("khmer_text", "").replace("\n", "\\N")
                    f.write(f"Dialogue: 0,{s},{e},Default,,0,0,0,,{txt}\n")

            total_video_duration = get_media_duration(str(video_in)) or 60.0

            clips_results = []
            for idx, d in enumerate(dialogues):
                it = d if isinstance(d, dict) else d.dict()
                txt = it.get("khmer_text", "").strip()
                if not txt:
                    continue
                v_id = "km-KH-SreymomNeural" if it.get("speaker") == "sreymom" else "km-KH-PisethNeural"
                r_wav = processed_dir / f"{ep_id}_r_{idx}.wav"
                f_wav = processed_dir / f"{ep_id}_f_{idx}.wav"

                if generate_khmer_audio_fast(txt, v_id, str(r_wav)):
                    fit_audio_exact_to_scene(str(r_wav), max(0.35, float(it["end"]) - float(it["start"])), str(f_wav))
                    clips_results.append({
                        "start": float(it["start"]),
                        "duration": get_media_duration(str(f_wav)),
                        "path": str(f_wav)
                    })

            user_db = load_user_db(db_file)
            user_db[ep_id]["progress"] = 70
            save_user_db(db_file, user_db)

            dub_track = processed_dir / f"{ep_id}_dub.wav"
            manifest_txt = processed_dir / f"{ep_id}_mf.txt"
            current_cursor = 0.0

            sil_master = processed_dir / "sil.wav"
            if not sil_master.exists():
                subprocess.run([FFMPEG_BIN, "-y", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", "30", "-c:a", "pcm_s16le", str(sil_master)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            with open(manifest_txt, "w", encoding="utf-8") as f_mf:
                for idx, c in enumerate(clips_results):
                    gap = c["start"] - current_cursor
                    if gap > 0.04:
                        s_chunk = processed_dir / f"{ep_id}_s_{idx}.wav"
                        subprocess.run([FFMPEG_BIN, "-y", "-i", str(sil_master), "-t", f"{gap:.3f}", "-c", "copy", str(s_chunk)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        f_mf.write(f"file '{str(s_chunk).replace(os.sep, '/')}'\n")
                    f_mf.write(f"file '{c['path'].replace(os.sep, '/')}'\n")
                    current_cursor = c["start"] + c["duration"]

                if total_video_duration > current_cursor:
                    e_chunk = processed_dir / f"{ep_id}_e.wav"
                    subprocess.run([FFMPEG_BIN, "-y", "-i", str(sil_master), "-t", f"{total_video_duration - current_cursor:.3f}", "-c", "copy", str(e_chunk)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    f_mf.write(f"file '{str(e_chunk).replace(os.sep, '/')}'\n")

            subprocess.run([FFMPEG_BIN, "-y", "-f", "concat", "-safe", "0", "-i", str(manifest_txt), "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2", str(dub_track)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            out_name = f"dubbed_{ep.get('code', 'ep')}_{uuid.uuid4().hex[:6]}.mp4"
            final_mp4 = processed_dir / out_name

            dub_vol = cfg_dict.get("dub_volume", 2.20)
            master_audio_filter = f"volume={dub_vol},alimiter=limit=0.98"

            cmd = [
                FFMPEG_BIN, "-y",
                "-i", str(video_in),
                "-i", str(dub_track),
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-filter:a", master_audio_filter,
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                str(final_mp4)
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            user_db = load_user_db(db_file)
            if ep_id in user_db:
                user_db[ep_id]["status"] = "completed"
                user_db[ep_id]["progress"] = 100
                user_db[ep_id]["output_url"] = f"/api/download/{active_sid}/{out_name}"
                save_user_db(db_file, user_db)
                completed_files.append(str(final_mp4))

@app.post("/api/batch-render")
async def start_batch_render(cfg: BatchRenderConfig, sid: Optional[str] = Cookie(None)):
    active_sid = cfg.session_id or sid or "default_user"
    worker = threading.Thread(target=render_worker, args=(active_sid, cfg.dict()), daemon=True)
    worker.start()
    return JSONResponse({"message": "Render started"})

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)