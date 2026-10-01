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

if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

try:
    import edge_tts
except ImportError:
    edge_tts = None

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

app = FastAPI(title="KhmerDub Studio Pro - True Lip-Sync Engine", version="160.0.0")

@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return Response(status_code=204)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE = BASE_DIR / "workspace"
UPLOADS_DIR = WORKSPACE / "uploads"
PROCESSED_DIR = WORKSPACE / "processed"
STATIC_DIR = BASE_DIR / "public"
DB_FILE = WORKSPACE / "episodes_db.json"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/outputs", StaticFiles(directory=str(PROCESSED_DIR)), name="outputs")

EPISODES_DB = {}
GLOBAL_WHISPER_INSTANCE = None
CURRENT_MODEL_NAME = None

def load_db():
    global EPISODES_DB
    if not isinstance(EPISODES_DB, dict):
        EPISODES_DB = {}

def save_db():
    try:
        with open(DB_FILE, "w", encoding="utf-8") as f:
            json.dump(EPISODES_DB, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[!] Save DB Error: {e}")

if DB_FILE.exists():
    try:
        DB_FILE.unlink()
    except Exception:
        pass

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
        cmd = [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
            str(file_path)
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
        val = res.stdout.strip()
        return float(val) if val else 0.0
    except Exception:
        return 0.0

def _run_edge_tts_isolated(text: str, voice: str, target_mp3: str) -> bool:
    async def _async_call():
        communicate = edge_tts.Communicate(text, voice, rate="+2%", pitch="+1Hz")
        await communicate.save(target_mp3)

    try:
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        new_loop.run_until_complete(_async_call())
        new_loop.close()
        return os.path.exists(target_mp3) and os.path.getsize(target_mp3) > 300
    except Exception:
        return False

# ==================== KHMER AUDIO GENERATION WITH MASTERING ====================
def generate_khmer_audio_sync(text: str, voice: str, out_wav_path: str) -> bool:
    clean_text = text.strip() or "បាទ"
    temp_mp3 = Path(out_wav_path).with_suffix(".mp3")
    success = False

    if edge_tts:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_edge_tts_isolated, clean_text, voice, str(temp_mp3))
            try:
                success = future.result(timeout=8.0)
            except Exception:
                success = False

    if not success:
        try:
            encoded = urllib.parse.quote(clean_text[:200])
            url = f"https://translate.google.com/translate_tts?ie=UTF-8&q={encoded}&tl=km&client=tw-ob"
            req = urllib.request.Request(
                url, 
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            )
            with urllib.request.urlopen(req, timeout=5) as resp, open(str(temp_mp3), "wb") as f:
                f.write(resp.read())
            if os.path.exists(temp_mp3) and os.path.getsize(temp_mp3) > 300:
                success = True
        except Exception:
            pass

    if success and os.path.exists(temp_mp3):
        studio_filter = (
            "highpass=f=90,"
            "lowpass=f=12000,"
            "equalizer=f=3000:t=q:w=1.5:g=3.5,"
            "equalizer=f=250:t=q:w=1.2:g=2.0,"
            "volume=1.8,"
            "alimiter=limit=0.95"
        )
        subprocess.run([
            "ffmpeg", "-y", "-i", str(temp_mp3),
            "-filter:a", studio_filter,
            "-ar", "48000",
            "-ac", "2",
            "-c:a", "pcm_s16le",
            str(out_wav_path)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            temp_mp3.unlink(missing_ok=True)
        except Exception:
            pass
        return os.path.exists(out_wav_path) and os.path.getsize(out_wav_path) > 300

    return False

# ==================== LIP-SYNC DYNAMIC TIME STRETCHING ====================
def sync_audio_duration_to_lips(raw_wav_path: str, target_duration: float, out_synced_path: str):
    """
    បម្លែងល្បឿនសំឡេងខ្មែរ (Atempo) ឱ្យត្រូវគ្នានឹងរយៈពេលមាត់តួអង្គកម្រិតមីលីវិនាទី
    """
    actual_dur = get_media_duration(raw_wav_path)
    if actual_dur <= 0.05 or target_duration <= 0.05:
        shutil.copyfile(raw_wav_path, out_synced_path)
        return

    ratio = actual_dur / target_duration
    # កំណត់កម្រិតល្បឿនឱ្យធម្មជាតិបំផុត (មិនឱ្យលឿនពេក ឬយឺតពេក)
    ratio = max(0.65, min(ratio, 1.85))

    # បង្កើត Filter Chain នៃ atempo (atempo ក្នុង FFmpeg គាំទ្ររវាង 0.5 ដល់ 2.0)
    filters = []
    curr = ratio
    while curr > 2.0:
        filters.append("atempo=2.0")
        curr /= 2.0
    while curr < 0.5:
        filters.append("atempo=0.5")
        curr /= 0.5
    filters.append(f"atempo={curr:.3f}")
    
    atempo_filter = ",".join(filters)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(raw_wav_path),
        "-filter:a", atempo_filter,
        "-ar", "48000",
        "-ac", "2",
        "-c:a", "pcm_s16le",
        str(out_synced_path)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.exists(out_synced_path) or os.path.getsize(out_synced_path) < 200:
        shutil.copyfile(raw_wav_path, out_synced_path)

def get_whisper_turbo_model(model_name: str = "large-v3-turbo"):
    global GLOBAL_WHISPER_INSTANCE, CURRENT_MODEL_NAME
    if GLOBAL_WHISPER_INSTANCE is None or CURRENT_MODEL_NAME != model_name:
        if WhisperModel:
            has_gpu = bool(shutil.which("nvidia-smi"))
            device = "cuda" if has_gpu else "cpu"
            comp_type = "float16" if has_gpu else "int8"
            try:
                GLOBAL_WHISPER_INSTANCE = WhisperModel(
                    model_name, 
                    device=device, 
                    compute_type=comp_type, 
                    cpu_threads=os.cpu_count() or 6,
                    num_workers=2
                )
                CURRENT_MODEL_NAME = model_name
            except Exception:
                GLOBAL_WHISPER_INSTANCE = WhisperModel("base", device="cpu", compute_type="int8")
                CURRENT_MODEL_NAME = "base"
    return GLOBAL_WHISPER_INSTANCE

def analyze_smart_speaker(orig_text: str, khmer_text: str, video_path: str, start_time: float, end_time: float, last_speaker: str) -> str:
    female_zh = r"(她|女人|女孩|小姐|妈妈|妻子|老婆|姐姐|妹妹|夫人|太太|娘|珊珊|小雪)"
    male_zh = r"(他|男人|男孩|先生|爸爸|丈夫|老公|哥哥|弟弟|少爷|老爹|林总|王总)"

    female_km = r"(នាង|នារី|ស្រី|ម៉ាក់|ម្ដាយ|បងស្រី|ប្អូនស្រី|ភរិយា|ប្រពន្ធ|កញ្ញា|អ្នកស្រី|អូន|ចៅស្រី|សានសាន)"
    male_km = r"(គាត់|បុរស|ប្រុស|ប៉ា|ឪពុក|បងប្រុស|ប្អូនប្រុស|ស្វាមី|ប្ដី|លោក|បង|ចៅប្រុស)"

    if re.search(female_zh, orig_text) or re.search(female_km, khmer_text):
        return "sreymom"
    if re.search(male_zh, orig_text) or re.search(male_km, khmer_text):
        return "piseth"

    if cv2:
        try:
            cap = cv2.VideoCapture(video_path)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            sample_times = [start_time + 0.2, (start_time + end_time) / 2.0, max(start_time, end_time - 0.2)]
            for t in sample_times:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.2, 4)
                if len(faces) > 0:
                    cap.release()
                    return "sreymom" if last_speaker == "piseth" else "piseth"
            cap.release()
        except Exception:
            pass

    return "sreymom" if last_speaker == "piseth" else "piseth"

def translate_drama_dialogue_to_khmer(text: str) -> str:
    clean_text = text.strip()
    if not clean_text:
        return ""
    clean_text = re.sub(r'^[（\(].*?[）\)]', '', clean_text)
    try:
        encoded = urllib.parse.quote(clean_text)
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=zh-CN&tl=km&dt=t&q={encoded}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data and data[0]:
                raw_km = "".join([s[0] for s in data[0] if s and s[0]]).strip()
                if raw_km:
                    cleaned_km = raw_km.replace("នារីម្នាក់", "នាង")\
                                       .replace("បុរសម្នាក់", "គាត់")\
                                       .replace("ជ្រើសរើសយក", "ជ្រើសយក")
                    return cleaned_km
    except Exception:
        pass
    return "អត្ថន័យសន្ទនា"

@app.get("/")
async def root_view():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Please place public/index.html correctly."}

@app.get("/api/download/{filename}")
async def download_file_direct(filename: str):
    file_path = PROCESSED_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="video/mp4"
    )

@app.post("/api/batch-upload")
async def batch_upload(files: List[UploadFile] = File(...)):
    load_db()
    uploaded_eps = []
    current_count = len(EPISODES_DB)
    for i, f in enumerate(files):
        ep_id = f"ep_{uuid.uuid4().hex[:6]}"
        ep_code = f"EP{current_count + i + 1:02d}"
        ext = Path(f.filename).suffix.lower() or ".mp4"
        dest_path = UPLOADS_DIR / f"{ep_id}_{ep_code}{ext}"

        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(f.file, buffer)

        EPISODES_DB[ep_id] = {
            "id": ep_id,
            "code": ep_code,
            "filename": f.filename,
            "path": str(dest_path),
            "status": "ready",
            "progress": 0,
            "dialogues": []
        }
        uploaded_eps.append(EPISODES_DB[ep_id])
    save_db()
    return {"uploaded": uploaded_eps}

@app.post("/api/clear-all-episodes")
async def clear_all_episodes():
    global EPISODES_DB
    EPISODES_DB.clear()
    save_db()
    for folder in [UPLOADS_DIR, PROCESSED_DIR]:
        for item in folder.glob("*"):
            try:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            except Exception:
                pass
    return {"message": "All episodes cleared successfully"}

@app.delete("/api/episodes/{episode_id}")
async def delete_single_episode_api(episode_id: str):
    global EPISODES_DB
    load_db()
    if episode_id in EPISODES_DB:
        ep = EPISODES_DB.pop(episode_id)
        try:
            if os.path.exists(ep.get("path", "")):
                os.remove(ep["path"])
        except Exception:
            pass

        new_db = {}
        for idx, (k, v) in enumerate(EPISODES_DB.items()):
            v["code"] = f"EP{idx + 1:02d}"
            new_db[k] = v
        EPISODES_DB = new_db
        save_db()
        return {"message": "Deleted successfully", "episodes": EPISODES_DB}
    raise HTTPException(status_code=404, detail="Episode not found")

# ==================== TRANSCRIBE WITH WORD-LEVEL TIMESTAMP ALIGNMENT ====================
@app.post("/api/transcribe-episode")
async def transcribe_episode(
    episode_id: str = Form(...),
    model_size: str = Form("large-v3-turbo"),
    lang: str = Form("zh"),
    speaker_mode: str = Form("auto")
):
    try:
        load_db()
        if episode_id not in EPISODES_DB:
            raise HTTPException(status_code=404, detail="Episode not found")

        ep = EPISODES_DB[episode_id]
        ep["status"] = "transcribing"
        video_path = ep["path"]
        audio_path = UPLOADS_DIR / f"{episode_id}_audio.wav"

        subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(audio_path)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        whisper = get_whisper_turbo_model(model_size)
        raw_segments = []
        if whisper:
            target_lang = "zh" if lang in ["zh", "chinese", "auto"] else lang
            # បើក word_timestamps=True ដើម្បីចាប់ពេលវេលាមាត់តួអង្គនិយាយឱ្យចំមីលីវិនាទី
            segments, _ = whisper.transcribe(
                str(audio_path),
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=250),
                language=target_lang,
                word_timestamps=True,
                beam_size=2,
                temperature=0.0,
                initial_prompt="这是一部现代短剧对话，包括男女主角的日常对话。"
            )
            raw_segments = list(segments)

        segments_payload = [
            {
                "id": idx + 1,
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "original_text": seg.text.strip()
            }
            for idx, seg in enumerate(raw_segments)
        ]

        dialogues = []
        last_speaker = "piseth"

        for orig in segments_payload:
            final_khmer = translate_drama_dialogue_to_khmer(orig["original_text"])
            
            if speaker_mode == "male":
                final_spk = "piseth"
            elif speaker_mode == "female":
                final_spk = "sreymom"
            else:
                final_spk = analyze_smart_speaker(
                    orig["original_text"],
                    final_khmer,
                    video_path,
                    orig["start"],
                    orig["end"],
                    last_speaker
                )

            last_speaker = final_spk

            dialogues.append({
                "id": orig["id"],
                "start": orig["start"],
                "end": orig["end"],
                "original_text": orig["original_text"],
                "khmer_text": final_khmer,
                "speaker": final_spk
            })

        ep["dialogues"] = dialogues
        ep["status"] = "transcribed"
        save_db()
        return JSONResponse({"episode_id": episode_id, "dialogues": dialogues})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/preview-tts")
async def preview_single_tts(
    text: str = Form(...), 
    speaker: str = Form("piseth")
):
    clean_text = text.strip() or "សួស្តី"
    voice = "km-KH-SreymomNeural" if speaker == "sreymom" else "km-KH-PisethNeural"
    uid = uuid.uuid4().hex[:6]
    out_wav = PROCESSED_DIR / f"prev_{uid}.wav"

    ok = generate_khmer_audio_sync(clean_text, voice, str(out_wav))
    if not ok or not out_wav.exists():
        raise HTTPException(status_code=500, detail="TTS Engine Failed")

    return FileResponse(path=str(out_wav), media_type="audio/wav", filename="preview.wav")

# ==================== RENDER WORKER WITH LIP-SYNC MATCHING ====================
def render_single_episode_sync(ep_id: str, cfg_dict: dict) -> str:
    load_db()
    ep = EPISODES_DB[ep_id]
    ep["status"] = "rendering"
    ep["progress"] = 15
    save_db()

    dialogues = []
    dialogues_map = cfg_dict.get("dialogues_map") or {}
    if ep_id in dialogues_map:
        dialogues = dialogues_map[ep_id]
        ep["dialogues"] = dialogues
    elif "dialogues" in ep and ep["dialogues"]:
        dialogues = ep["dialogues"]

    video_in = ep["path"]
    ass_file = PROCESSED_DIR / f"{ep_id}.ass"

    res_w, res_h = (1080, 1920) if cfg_dict.get("aspect_ratio") != "16:9" else (1920, 1080)
    font_size = 50
    if cfg_dict.get("resolution") == "4k":
        res_w, res_h = (2160, 3840) if cfg_dict.get("aspect_ratio") != "16:9" else (3840, 2160)
        font_size = 94

    with open(ass_file, "w", encoding="utf-8") as f:
        f.write(f"[Script Info]\nTitle: KhmerDub Engine\nScriptType: v4.00+\nPlayResX: {res_w}\nPlayResY: {res_h}\n\n")
        f.write("[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, Italic, Alignment, MarginV, Outline, Shadow\n")
        f.write(f"Style: Default,Khmer OS Siemreap,{font_size},{cfg_dict.get('sub_color', '&H00FFFF')},&H00000000,&H80000000,1,0,2,85,2.5,1.2\n\n")
        f.write("[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        for d in dialogues:
            it = d if isinstance(d, dict) else d.dict()
            s = format_ass_time(it["start"])
            e = format_ass_time(it["end"])
            txt = it.get("khmer_text", "").replace("\n", "\\N")
            f.write(f"Dialogue: 0,{s},{e},Default,,0,0,0,,{txt}\n")

    ep["progress"] = 30
    save_db()

    clips = []
    total_d = len(dialogues)
    for i, d in enumerate(dialogues):
        it = d if isinstance(d, dict) else d.dict()
        txt = it.get("khmer_text", "").strip()
        if not txt:
            continue

        spk = it.get("speaker", "piseth")
        voice_id = "km-KH-SreymomNeural" if spk == "sreymom" else "km-KH-PisethNeural"

        raw_line_wav = PROCESSED_DIR / f"{ep_id}_raw_{i}.wav"
        synced_line_wav = PROCESSED_DIR / f"{ep_id}_sync_{i}.wav"

        ok = generate_khmer_audio_sync(txt, voice_id, str(raw_line_wav))
        if ok and raw_line_wav.exists() and os.path.getsize(raw_line_wav) > 300:
            target_scene_duration = float(it["end"]) - float(it["start"])
            # អនុវត្ត Dynamic Lip-Sync Time-Stretching ដើម្បីឱ្យសំឡេងត្រូវគ្នានឹងមាត់តួអង្គ
            sync_audio_duration_to_lips(str(raw_line_wav), target_scene_duration, str(synced_line_wav))
            dur = get_media_duration(str(synced_line_wav))
            clips.append({"start": float(it["start"]), "path": str(synced_line_wav), "duration": dur})

        if total_d > 0:
            ep["progress"] = 30 + int((i / total_d) * 35)
            save_db()

    full_video_duration = get_media_duration(str(video_in))
    if full_video_duration <= 0:
        full_video_duration = 300.0

    ep["progress"] = 70
    save_db()

    dub_track = PROCESSED_DIR / f"{ep_id}_dub.wav"
    concat_list_file = PROCESSED_DIR / f"{ep_id}_concat_manifest.txt"
    current_cursor = 0.0

    with open(concat_list_file, "w", encoding="utf-8") as f_list:
        for idx, clip in enumerate(clips):
            start_time = clip["start"]
            gap = start_time - current_cursor

            if gap > 0.04:
                silence_file = PROCESSED_DIR / f"{ep_id}_sil_{idx}.wav"
                subprocess.run([
                    "ffmpeg", "-y", "-f", "lavfi",
                    "-i", "anullsrc=r=48000:cl=stereo",
                    "-t", f"{gap:.3f}",
                    "-c:a", "pcm_s16le",
                    str(silence_file)
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                f_list.write(f"file '{str(silence_file).replace(os.sep, '/')}'\n")

            f_list.write(f"file '{clip['path'].replace(os.sep, '/')}'\n")
            current_cursor = start_time + clip["duration"]

        if full_video_duration > current_cursor:
            end_gap = full_video_duration - current_cursor
            end_silence = PROCESSED_DIR / f"{ep_id}_end_sil.wav"
            subprocess.run([
                "ffmpeg", "-y", "-f", "lavfi",
                "-i", "anullsrc=r=48000:cl=stereo",
                "-t", f"{end_gap:.3f}",
                "-c:a", "pcm_s16le",
                str(end_silence)
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            f_list.write(f"file '{str(end_silence).replace(os.sep, '/')}'\n")

    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_list_file),
        "-c:a", "pcm_s16le",
        "-ar", "48000",
        "-ac", "2",
        str(dub_track)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    ep["progress"] = 85
    save_db()

    render_token = uuid.uuid4().hex[:6]
    out_name = f"dubbed_{ep.get('code', 'ep')}_{cfg_dict.get('resolution', '1080p')}_{render_token}.mp4"
    final_mp4 = PROCESSED_DIR / out_name

    ass_path_posix = ass_file.as_posix()
    if sys.platform == "win32" and ":" in ass_path_posix:
        drive, path_part = ass_path_posix.split(":", 1)
        sub_filter = f"subtitles='{drive}\\:{path_part}'"
    else:
        sub_filter = f"subtitles='{ass_path_posix}'"

    trim_args = []
    if cfg_dict.get("trim_in", 0.0) > 0:
        trim_args.extend(["-ss", f"{cfg_dict['trim_in']:.2f}"])
    if cfg_dict.get("trim_out") and cfg_dict["trim_out"] > cfg_dict.get("trim_in", 0.0):
        trim_args.extend(["-to", f"{cfg_dict['trim_out']:.2f}"])

    dub_vol = cfg_dict.get("dub_volume", 2.20)
    master_audio_filter = f"volume={dub_vol},alimiter=limit=0.98"

    target_w, target_h = (1080, 1920) if cfg_dict.get("aspect_ratio") != "16:9" else (1920, 1080)
    scale_filter = f"scale={target_w}:{target_h}:flags=lanczos:force_original_aspect_ratio=decrease,pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2"
    combined_vf = f"{scale_filter},{sub_filter}"

    print(f"[*] Rendering Full HD 1080p with Lip-Sync to: {out_name}...")
    cmd = [
        "ffmpeg", "-y",
        *trim_args,
        "-i", str(video_in),
        "-i", str(dub_track),
        "-vf", combined_vf,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-filter:a", master_audio_filter,
        "-c:v", "libx264",
        "-preset", "faster",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "320k",
        "-ar", "48000",
        str(final_mp4)
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors='replace')

    if res.returncode != 0 or not final_mp4.exists() or os.path.getsize(final_mp4) < 1000:
        cmd_fb = [
            "ffmpeg", "-y",
            *trim_args,
            "-i", str(video_in),
            "-i", str(dub_track),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-filter:a", master_audio_filter,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "320k",
            "-ar", "48000",
            str(final_mp4)
        ]
        subprocess.run(cmd_fb, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    ep["status"] = "completed"
    ep["progress"] = 100
    ep["output_url"] = f"/api/download/{out_name}"
    save_db()
    print(f"[✓] Lip-Sync render completed successfully: {out_name}\n")
    return str(final_mp4)

def run_render_in_daemon_thread(cfg_dict: dict):
    completed_files = []
    ep_ids = cfg_dict.get("episode_ids", [])
    for ep_id in ep_ids:
        load_db()
        if ep_id in EPISODES_DB:
            try:
                out_file = render_single_episode_sync(ep_id, cfg_dict)
                completed_files.append(out_file)
            except Exception as e:
                print(f"[!] Error rendering {ep_id}: {e}")
                EPISODES_DB[ep_id]["status"] = "failed"
                save_db()

    if cfg_dict.get("merge_all_into_one") and len(completed_files) > 1:
        merged_name = f"merged_full_{cfg_dict.get('resolution', '1080p')}_{uuid.uuid4().hex[:6]}.mp4"
        merged_path = PROCESSED_DIR / merged_name
        list_txt = PROCESSED_DIR / "concat_list.txt"
        with open(list_txt, "w", encoding="utf-8") as f:
            for c in completed_files:
                f.write(f"file '{c.replace(os.sep, '/')}'\n")

        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_txt),
            "-c", "copy", str(merged_path)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        EPISODES_DB["merged_master"] = {
            "id": "merged_master",
            "code": "FULL_MERGED",
            "filename": merged_name,
            "status": "completed",
            "progress": 100,
            "output_url": f"/api/download/{merged_name}"
        }
        save_db()

@app.post("/api/batch-render")
def start_batch_render(cfg: BatchRenderConfig):
    cfg_data = cfg.dict()
    worker = threading.Thread(target=run_render_in_daemon_thread, args=(cfg_data,), daemon=True)
    worker.start()
    return JSONResponse({"message": "Turbo Batch rendering started in isolated daemon thread"})

@app.get("/api/episodes")
def get_episodes():
    load_db()
    return JSONResponse(EPISODES_DB)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    bind_host = "0.0.0.0" if os.environ.get("RENDER") else "127.0.0.1"
    
    print("\n" + "=" * 55)
    print(f"🚀 KhmerDub Studio Pro is READY!")
    print(f"👉 Link: http://127.0.0.1:{port}")
    print("=" * 55 + "\n")

    uvicorn.run("server:app", host=bind_host, port=port, reload=False)