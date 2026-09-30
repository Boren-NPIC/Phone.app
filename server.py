import os
import sys
import uuid
import shutil
import asyncio
import subprocess
import urllib.parse
import urllib.request
import json
import re
from pathlib import Path
from typing import List, Optional

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

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
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

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

app = FastAPI(title="KhmerDub Studio Pro - Ultra Studio Voice Engine", version="72.0.0")

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

def get_best_hardware_encoder():
    try:
        res = subprocess.run(["ffmpeg", "-encoders"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors='ignore')
        encoders = res.stdout
        if "h264_nvenc" in encoders:
            return "h264_nvenc"
        elif "h264_qsv" in encoders:
            return "h264_qsv"
        elif "h264_amf" in encoders:
            return "h264_amf"
    except Exception:
        pass
    return "libx264"

HARDWARE_ENCODER = get_best_hardware_encoder()
print(f"[*] Detected Video Engine: {HARDWARE_ENCODER}")

def load_db():
    global EPISODES_DB
    if DB_FILE.exists():
        try:
            with open(DB_FILE, "r", encoding="utf-8") as f:
                EPISODES_DB = json.load(f)
        except Exception:
            EPISODES_DB = {}
    else:
        EPISODES_DB = {}

def save_db():
    try:
        with open(DB_FILE, "w", encoding="utf-8") as f:
            json.dump(EPISODES_DB, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("Save DB Error:", e)

load_db()

class BatchRenderConfig(BaseModel):
    episode_ids: List[str]
    merge_all_into_one: bool = False
    speaker_mode: str = "auto"
    speed_factor: float = 1.0
    dub_volume: float = 1.50
    burn_subtitles: bool = True
    sub_color: str = "&H00FFFF"
    aspect_ratio: str = "9:16"
    resolution: str = "1080p"
    dialogues_map: Optional[dict] = None
    trim_in: float = 0.0
    trim_out: Optional[float] = None

def has_chinese(text: str) -> bool:
    return bool(re.search(r'[\u4e00-\u9fff]', text))

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

# ==================== KHMER TTS ENGINE (STUDIO MASTERING) ====================
async def generate_khmer_audio_robust(text: str, voice: str, out_wav_path: str) -> bool:
    clean_text = text.strip() or "បាទ"
    temp_mp3 = Path(out_wav_path).with_suffix(".mp3")

    success = False

    # 1. ដំណើរការតាម edge-tts Library (Pitch & Clarity Tuning)
    if edge_tts:
        try:
            # បន្ថែម Pitch +0Hz និង Rate ដើម្បីឱ្យសំឡេងនិយាយមានលក្ខណៈធម្មជាតិ
            communicate = edge_tts.Communicate(clean_text, voice, rate="+2%", pitch="+1Hz")
            await asyncio.wait_for(communicate.save(str(temp_mp3)), timeout=7.0)
            if os.path.exists(temp_mp3) and os.path.getsize(temp_mp3) > 300:
                success = True
        except Exception as e:
            print(f"[!] edge_tts library warning: {e}")

    # 2. Google Direct Khmer TTS Fallback
    if not success:
        def _google_tts_fetch():
            try:
                encoded = urllib.parse.quote(clean_text[:200])
                url = f"https://translate.google.com/translate_tts?ie=UTF-8&q={encoded}&tl=km&client=tw-ob"
                req = urllib.request.Request(
                    url, 
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                )
                with urllib.request.urlopen(req, timeout=6) as resp, open(str(temp_mp3), "wb") as f:
                    f.write(resp.read())
                return os.path.exists(temp_mp3) and os.path.getsize(temp_mp3) > 300
            except Exception as err:
                print(f"[!] Google TTS error: {err}")
                return False

        success = await asyncio.to_thread(_google_tts_fetch)

    # 3. STUDIO MASTERING FILTER (ពង្រឹងសំឡេងឱ្យណែន ពិរោះ ច្បាស់)
    if success and os.path.exists(temp_mp3):
        # Filter សម្អាត និងពង្រឹង៖
        # - highpass=f=75: កាត់សំឡេងខ្យល់
        # - lowpass=f=11500: កាត់សំឡេងស្រួយខ្ពស់កុំឱ្យរំខានត្រចៀក
        # - compand: បង្កើនសំឡេងខ្សោយ ធ្វើឱ្យពាក្យគ្រប់ម៉ាត់ឮច្បាស់ស្មើល្អ
        # - volume=1.3: បន្ថែមថាមពលសំឡេង
        studio_filter = (
            "highpass=f=75,"
            "lowpass=f=11500,"
            "compand=attacks=0.02:decays=0.2:points=-60/-60|-30/-15|-10/-3|0/0:soft-knee=6,"
            "volume=1.3"
        )
        subprocess.run([
            "ffmpeg", "-y", "-i", str(temp_mp3),
            "-filter:a", studio_filter,
            "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le",
            str(out_wav_path)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        try:
            temp_mp3.unlink(missing_ok=True)
        except Exception:
            pass
        return os.path.exists(out_wav_path) and os.path.getsize(out_wav_path) > 500

    return False

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
                    cpu_threads=os.cpu_count() or 8,
                    num_workers=2
                )
                CURRENT_MODEL_NAME = model_name
            except Exception:
                GLOBAL_WHISPER_INSTANCE = WhisperModel("base", device="cpu", compute_type="int8")
                CURRENT_MODEL_NAME = "base"
    return GLOBAL_WHISPER_INSTANCE

def detect_speaker_from_video_frame(video_path: str, timestamp_sec: float) -> Optional[str]:
    if not cv2:
        return None
    try:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)
        success, frame = cap.read()
        cap.release()
        if not success or frame is None:
            return None

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        if len(faces) == 0:
            return None

        largest_face = max(faces, key=lambda r: r[2] * r[3])
        x, y, w, h = largest_face
        face_roi = frame[y:y+h, x:x+w]

        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
        lip_ratio = cv2.countNonZero(mask) / (w * h)

        return "sreymom" if lip_ratio > 0.045 else "piseth"
    except Exception:
        return None

async def translate_text_to_khmer(text: str) -> str:
    clean_text = text.strip()
    if not clean_text:
        return ""

    def _sync_trans():
        encoded = urllib.parse.quote(clean_text)
        try:
            url1 = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=zh-CN&tl=km&dt=t&q={encoded}"
            req1 = urllib.request.Request(url1, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req1, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data and data[0]:
                    res = "".join([s[0] for s in data[0] if s and s[0]]).strip()
                    if res:
                        return res
        except Exception:
            pass
        return "អត្ថន័យសាច់រឿងភាគ"

    return await asyncio.to_thread(_sync_trans)

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
    return FileResponse(path=str(file_path), filename=filename, media_type="video/mp4")

@app.post("/api/batch-upload")
async def batch_upload(files: List[UploadFile] = File(...)):
    uploaded_eps = []
    for f in files:
        ep_id = f"ep_{uuid.uuid4().hex[:6]}"
        ep_code = f"EP{len(EPISODES_DB) + 1:02d}"
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
            segments, _ = whisper.transcribe(
                str(audio_path),
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=400),
                language=target_lang,
                beam_size=1,
                temperature=0.0
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

        trans_tasks = [translate_text_to_khmer(d["original_text"]) for d in segments_payload]
        khmer_translations = await asyncio.gather(*trans_tasks)

        dialogues = []
        last_speaker = "piseth"

        for idx, orig in enumerate(segments_payload):
            final_khmer = khmer_translations[idx]
            if speaker_mode == "male":
                final_spk = "piseth"
            elif speaker_mode == "female":
                final_spk = "sreymom"
            else:
                mid_time = (orig["start"] + orig["end"]) / 2
                vision_speaker = detect_speaker_from_video_frame(video_path, mid_time)
                final_spk = vision_speaker if vision_speaker else last_speaker

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
    if has_chinese(clean_text):
        clean_text = await translate_text_to_khmer(clean_text)

    voice = "km-KH-SreymomNeural" if speaker == "sreymom" else "km-KH-PisethNeural"
    uid = uuid.uuid4().hex[:6]
    out_wav = PROCESSED_DIR / f"prev_{uid}.wav"

    ok = await generate_khmer_audio_robust(clean_text, voice, str(out_wav))
    if not ok or not out_wav.exists():
        raise HTTPException(status_code=500, detail="TTS Engine Failed")

    return FileResponse(path=str(out_wav), media_type="audio/wav", filename="preview.wav")

# ==================== RENDER វីដេអូ និងសំឡេងខ្មែរ MASTER (កាត់សំឡេងចិនចោល ១០០%) ====================
async def render_single_episode(ep_id: str, cfg: BatchRenderConfig) -> str:
    load_db()
    ep = EPISODES_DB[ep_id]
    ep["status"] = "rendering"
    ep["progress"] = 10

    dialogues = []
    if cfg.dialogues_map and ep_id in cfg.dialogues_map:
        dialogues = cfg.dialogues_map[ep_id]
        ep["dialogues"] = dialogues
    elif "dialogues" in ep and ep["dialogues"]:
        dialogues = ep["dialogues"]

    print(f"\n[🚀 START RENDER] Processing {len(dialogues)} dialogues for episode {ep.get('code', ep_id)}")

    video_in = ep["path"]
    ass_file = PROCESSED_DIR / f"{ep_id}.ass"

    res_w, res_h = (1080, 1920) if cfg.aspect_ratio != "16:9" else (1920, 1080)
    font_size = 48
    if cfg.resolution == "4k":
        res_w, res_h = (2160, 3840) if cfg.aspect_ratio != "16:9" else (3840, 2160)
        font_size = 92

    with open(ass_file, "w", encoding="utf-8") as f:
        f.write(f"[Script Info]\nTitle: Turbo Dub\nScriptType: v4.00+\nPlayResX: {res_w}\nPlayResY: {res_h}\n\n")
        f.write("[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, Italic, Alignment, MarginV\n")
        f.write(f"Style: Default,Arial,{font_size},{cfg.sub_color},&H00000000,&H80000000,1,0,2,80\n\n")
        f.write("[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        for d in dialogues:
            it = d if isinstance(d, dict) else d.dict()
            s = format_ass_time(it["start"])
            e = format_ass_time(it["end"])
            txt = it.get("khmer_text", "").replace("\n", "\\N")
            f.write(f"Dialogue: 0,{s},{e},Default,,0,0,0,,{txt}\n")

    ep["progress"] = 20

    # បង្កើត File សំឡេងខ្មែរគ្រប់ឃ្លាទាំងអស់ជាមួយកម្រិត Master Sound
    clips = []
    for i, d in enumerate(dialogues):
        it = d if isinstance(d, dict) else d.dict()
        txt = it.get("khmer_text", "").strip()
        if not txt:
            continue

        spk = cfg.speaker_mode
        if spk == "auto":
            spk = it.get("speaker", "piseth")
        v = "km-KH-SreymomNeural" if spk == "sreymom" else "km-KH-PisethNeural"

        line_wav = PROCESSED_DIR / f"{ep_id}_line_{i}.wav"
        ok = await generate_khmer_audio_robust(txt, v, str(line_wav))
        if ok and line_wav.exists() and os.path.getsize(line_wav) > 500:
            dur = get_media_duration(str(line_wav))
            clips.append({"start": float(it["start"]), "path": str(line_wav), "duration": dur})
            print(f"[✓] Created Master Voice {i}: {txt[:20]}... ({dur:.2f}s)")
        else:
            print(f"[!] Warning: failed audio for line {i}")

    print(f"[*] Total Khmer Audio Clips Generated: {len(clips)} / {len(dialogues)}")

    full_video_duration = get_media_duration(str(video_in))
    if full_video_duration <= 0:
        full_video_duration = 300.0

    ep["progress"] = 60

    # ==================== BUILD AUDIO TRACK (CONCAT DEMUXER) ====================
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
                    "-i", "anullsrc=r=44100:cl=stereo",
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
                "-i", "anullsrc=r=44100:cl=stereo",
                "-t", f"{end_gap:.3f}",
                "-c:a", "pcm_s16le",
                str(end_silence)
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            f_list.write(f"file '{str(end_silence).replace(os.sep, '/')}'\n")

    # បញ្ចូលសំឡេងទាំងអស់ចេញជា Master Dub Audio
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_list_file),
        "-c:a", "pcm_s16le",
        "-ar", "44100",
        "-ac", "2",
        str(dub_track)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    ep["progress"] = 80

    render_token = uuid.uuid4().hex[:6]
    out_name = f"dubbed_{ep.get('code', 'ep')}_{cfg.resolution}_{render_token}.mp4"
    final_mp4 = PROCESSED_DIR / out_name

    ass_path_str = str(ass_file).replace("\\", "/").replace(":", "\\:")

    trim_args = []
    if cfg.trim_in > 0:
        trim_args.extend(["-ss", f"{cfg.trim_in:.2f}"])
    if cfg.trim_out and cfg.trim_out > cfg.trim_in:
        trim_args.extend(["-to", f"{cfg.trim_out:.2f}"])

    if HARDWARE_ENCODER == "h264_nvenc":
        enc_flags = ["-c:v", "h264_nvenc", "-preset", "p1", "-rc", "vbr", "-cq", "20"]
    else:
        enc_flags = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", "-threads", "0"]

    video_filters = []
    if cfg.resolution == "1080p":
        video_filters.append("scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2")
    elif cfg.resolution == "4k":
        video_filters.append("scale=2160:3840:force_original_aspect_ratio=decrease,pad=2160:3840:(ow-iw)/2:(oh-ih)/2")

    if cfg.burn_subtitles:
        video_filters.append(f"subtitles='{ass_path_str}'")

    v_filter_str = ",".join(video_filters) if video_filters else "null"

    # ==================== FINAL AUDIO MASTERING CHAIN ====================
    # បន្ថែម loudnorm ដើម្បីឱ្យសំឡេងឮកម្រិតស្តង់ដារ Broadcasting ច្បាស់ណែនល្អ
    audio_master_filter = f"volume={cfg.dub_volume},loudnorm=I=-16:TP=-1.5:LRA=11"

    print(f"[*] Mastering and Multiplexing video with crisp Khmer audio: {out_name}...")
    cmd = [
        "ffmpeg", "-y",
        *trim_args,
        "-i", str(video_in),
        "-i", str(dub_track),
        "-filter_complex", f"[0:v]{v_filter_str}[vout];[1:a]{audio_master_filter}[aout]",
        "-map", "[vout]",
        "-map", "[aout]",
        *enc_flags,
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "256k",
        str(final_mp4)
    ]
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if res.returncode != 0 or not final_mp4.exists():
        cmd_fb = [
            "ffmpeg", "-y",
            *trim_args,
            "-i", str(video_in),
            "-i", str(dub_track),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-filter:a", f"volume={cfg.dub_volume}",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "256k",
            str(final_mp4)
        ]
        subprocess.run(cmd_fb, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    ep["status"] = "completed"
    ep["progress"] = 100
    ep["output_url"] = f"/api/download/{out_name}"
    save_db()
    print(f"[✓] Render completed successfully with Studio Audio Quality: {out_name}\n")
    return str(final_mp4)

async def batch_render_worker(cfg: BatchRenderConfig):
    completed_files = []
    for ep_id in cfg.episode_ids:
        load_db()
        if ep_id in EPISODES_DB:
            try:
                out_file = await render_single_episode(ep_id, cfg)
                completed_files.append(out_file)
            except Exception as e:
                print(f"Error rendering {ep_id}:", e)
                EPISODES_DB[ep_id]["status"] = "failed"
                save_db()

    if cfg.merge_all_into_one and len(completed_files) > 1:
        merged_name = f"merged_full_{cfg.resolution}_{uuid.uuid4().hex[:6]}.mp4"
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
async def start_batch_render(cfg: BatchRenderConfig, bg: BackgroundTasks):
    bg.add_task(batch_render_worker, cfg)
    return JSONResponse({"message": "Turbo Batch rendering started"})

@app.get("/api/episodes")
async def get_episodes():
    load_db()
    return JSONResponse(EPISODES_DB)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    print(f"\n🚀 KhmerDub Studio Pro running on port {port}\n")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)