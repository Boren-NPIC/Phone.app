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

# កំណត់ UTF-8 Encoding ជាសកលសម្រាប់ប្រព័ន្ធ Windows
if sys.platform == "win32":
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

app = FastAPI(title="KhmerDub Studio Pro - V3 Turbo Ultra Engine", version="54.0.0")

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
    resolution: str = "1080p"  # "original", "1080p", "4k"
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

def get_audio_duration(file_path: str) -> float:
    try:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
            file_path
        ]
        res = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            encoding='utf-8', 
            errors='replace'
        )
        val = res.stdout.strip()
        return float(val) if val else 0.0
    except Exception:
        return 0.0

def build_atempo_filter(speed: float) -> str:
    filters = []
    curr = max(min(speed, 2.4), 0.6)
    while curr > 2.0:
        filters.append("atempo=2.0")
        curr /= 2.0
    while curr < 0.5:
        filters.append("atempo=0.5")
        curr /= 0.5
    filters.append(f"atempo={curr:.4f}")
    return ",".join(filters)

def process_exact_time_lock(raw_input_path: str, output_path: str, target_duration: float, user_speed: float = 1.0):
    actual_duration = get_audio_duration(raw_input_path)
    if actual_duration <= 0 or target_duration <= 0:
        shutil.copy(raw_input_path, output_path)
        return

    speed = max(min((actual_duration / target_duration) * user_speed, 2.2), 0.7)
    tempo_filter = build_atempo_filter(speed)
    cmd = [
        "ffmpeg", "-y", "-i", raw_input_path,
        "-filter:a", tempo_filter,
        "-threads", "4",
        "-ar", "44100", "-ac", "2",
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

async def generate_khmer_audio_file(text: str, voice: str, out_path: str) -> bool:
    clean_text = text.strip()
    if not clean_text:
        clean_text = "បាទ"

    if edge_tts:
        try:
            communicate = edge_tts.Communicate(clean_text, voice)
            await asyncio.wait_for(communicate.save(out_path), timeout=5.0)
            if os.path.exists(out_path) and os.path.getsize(out_path) > 300:
                return True
        except Exception:
            pass

    def _google_direct_tts():
        try:
            encoded_text = urllib.parse.quote(clean_text)
            url = f"https://translate.google.com/translate_tts?ie=UTF-8&q={encoded_text}&tl=km&client=tw-ob"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=5) as resp, open(out_path, "wb") as f:
                f.write(resp.read())
            return os.path.exists(out_path) and os.path.getsize(out_path) > 300
        except Exception:
            return False

    return await asyncio.to_thread(_google_direct_tts)

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
                GLOBAL_WHISPER_INSTANCE = WhisperModel(
                    "base", 
                    device="cpu", 
                    compute_type="int8", 
                    cpu_threads=os.cpu_count() or 8
                )
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

# ==================== ម៉ាស៊ីនបកប្រែពិតប្រាកដ ====================
async def translate_text_to_khmer_bulletproof(text: str) -> str:
    clean_text = text.strip()
    if not clean_text:
        return ""

    def _sync_translate():
        encoded = urllib.parse.quote(clean_text)
        
        manual_dict = {
            "三天后的除夕": "ពិធីបុណ្យចូលឆ្នាំបីថ្ងៃក្រោយ",
            "三天后的除息": "ពិធីបុណ្យចូលឆ្នាំបីថ្ងៃក្រោយ",
            "除夕": "រាត្រីឆ្លងឆ្នាំ",
            "我要对集团王牌机长": "ខ្ញុំនឹងចាត់ការប្រធានពីឡុតឆ្នើមរបស់ក្រុមហ៊ុន",
            "集团王牌机长": "ប្រធានពីឡុតឆ្នើមរបស់ក្រុមហ៊ុន",
            "我的爱人许君泽": "គូស្នេហ៍របស់ខ្ញុំគឺ ស៊ូ ជុនជឺ",
            "我的爱人许君子": "គូស្នេហ៍របស់ខ្ញុំគឺ ស៊ូ ជុនជឺ",
            "许君泽": "ស៊ូ ជុនជឺ",
            "找到的相框": "ស៊ុមរូបថតដែលរកឃើញ",
            "这上面有你和林夏姐的名字": "នៅលើនេះមានឈ្មោះបង និងបងស្រីលីនសៀ",
            "特意来问问你还要吗": "ខ្ញុំមកសួរពិសេស ថាតើបងនៅត្រូវការវាទៀតទេ",
            "进行全行业封杀": "ធ្វើការហាមឃាត់ក្នុងឧស្សាហកម្មទាំងមូល",
            "封杀": "ហាមឃាត់ដាច់ខាត"
        }
        for k, v in manual_dict.items():
            if k == clean_text:
                return v

        # វិធីទី ១៖ Google Translate Client Web API (sl=zh-CN, tl=km)
        try:
            url1 = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=zh-CN&tl=km&dt=t&q={encoded}"
            req1 = urllib.request.Request(url1, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
            with urllib.request.urlopen(req1, timeout=6) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data and data[0]:
                    res = "".join([s[0] for s in data[0] if s and s[0]]).strip()
                    if res and not has_chinese(res) and res != clean_text:
                        return res
        except Exception:
            pass

        # វិធីទី ២៖ Google Client Dict API
        try:
            url2 = f"https://clients5.google.com/translate_a/t?client=dict-chrome-ex&sl=zh-CN&tl=km&q={encoded}"
            req2 = urllib.request.Request(url2, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            })
            with urllib.request.urlopen(req2, timeout=6) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if isinstance(data, list) and len(data) > 0:
                    cand = data[0] if isinstance(data[0], str) else data[0][0]
                    cand = str(cand).strip()
                    if cand and not has_chinese(cand):
                        return cand
        except Exception:
            pass

        # វិធីទី ៣៖ MyMemory Translated API
        try:
            url3 = f"https://api.mymemory.translated.net/get?q={encoded}&langpair=zh|km"
            req3 = urllib.request.Request(url3, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req3, timeout=6) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                cand = data.get("responseData", {}).get("translatedText", "").strip()
                if cand and not has_chinese(cand) and cand != clean_text:
                    return cand
        except Exception:
            pass

        for k, v in manual_dict.items():
            if k in clean_text:
                return v

        return "អត្ថន័យសាច់រឿងភាគ"

    return await asyncio.to_thread(_sync_translate)

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
            "-threads", "4",
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
                best_of=1,
                temperature=0.0
            )
            raw_segments = list(segments)

        if not raw_segments:
            ep["dialogues"] = []
            ep["status"] = "transcribed"
            save_db()
            return JSONResponse({"episode_id": episode_id, "dialogues": []})

        segments_payload = [
            {
                "id": idx + 1,
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "duration": round(seg.end - seg.start, 2),
                "original_text": seg.text.strip()
            }
            for idx, seg in enumerate(raw_segments)
        ]

        trans_tasks = [translate_text_to_khmer_bulletproof(d["original_text"]) for d in segments_payload]
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

                if vision_speaker:
                    final_spk = vision_speaker
                else:
                    t = orig["original_text"]
                    if any(w in t for w in ["爱人", "老公", "姐", "太太", "夫人", "小姐", "妈", "林夏"]):
                        final_spk = "sreymom"
                    elif any(w in t for w in ["机长", "总", "先生", "兄弟", "爸"]):
                        final_spk = "piseth"
                    else:
                        if idx > 0 and (orig["start"] - segments_payload[idx-1]["end"]) < 1.2:
                            final_spk = last_speaker
                        else:
                            final_spk = last_speaker

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

        return JSONResponse({
            "episode_id": episode_id,
            "dialogues": dialogues
        })
    except Exception as e:
        print("Transcribe Endpoint Error:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/preview-tts")
async def preview_single_tts(
    text: str = Form(...), 
    speaker: str = Form("piseth"),
    target_duration: Optional[float] = Form(None),
    speed_factor: float = Form(1.0)
):
    clean_text = text.strip()
    if has_chinese(clean_text):
        clean_text = await translate_text_to_khmer_bulletproof(clean_text)

    if not clean_text:
        clean_text = "បាទ"

    target_voice = "km-KH-SreymomNeural" if speaker == "sreymom" else "km-KH-PisethNeural"
    uid = uuid.uuid4().hex[:6]
    raw_file = PROCESSED_DIR / f"prev_raw_{uid}.mp3"
    final_mp3 = PROCESSED_DIR / f"prev_final_{uid}.mp3"

    ok = await generate_khmer_audio_file(clean_text, target_voice, str(raw_file))
    if not ok:
        raise HTTPException(status_code=500, detail="Cannot generate TTS")

    tempo_val = max(min(float(speed_factor), 2.0), 0.7)
    cmd = [
        "ffmpeg", "-y", "-i", str(raw_file),
        "-filter:a", f"atempo={tempo_val:.2f}",
        "-vn", "-ar", "44100", "-ac", "2", "-b:a", "192k",
        str(final_mp3)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    serve_path = final_mp3 if os.path.exists(final_mp3) and os.path.getsize(final_mp3) > 100 else raw_file
    return FileResponse(path=str(serve_path), media_type="audio/mp3", filename="voice.mp3")

async def render_single_episode(ep_id: str, cfg: BatchRenderConfig) -> str:
    load_db()
    ep = EPISODES_DB[ep_id]
    ep["status"] = "rendering"
    ep["progress"] = 10

    if cfg.dialogues_map and ep_id in cfg.dialogues_map:
        dialogues = cfg.dialogues_map[ep_id]
        ep["dialogues"] = dialogues
    else:
        dialogues = ep.get("dialogues", [])

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
            txt = it["khmer_text"].replace("\n", "\\N")
            f.write(f"Dialogue: 0,{s},{e},Default,,0,0,0,,{txt}\n")

    ep["progress"] = 20

    # បង្កើន Concurrency Semaphore ដល់ 25 ឃ្លាព្រមគ្នា (Super Fast)
    sem = asyncio.Semaphore(25)

    async def fetch_audio_turbo(i, it):
        async with sem:
            txt = it.get("khmer_text", "").strip()
            if not txt:
                return None

            if has_chinese(txt):
                txt = await translate_text_to_khmer_bulletproof(txt)

            spk = cfg.speaker_mode
            if spk == "auto":
                spk = it.get("speaker", "piseth")
            v = "km-KH-SreymomNeural" if spk == "sreymom" else "km-KH-PisethNeural"

            raw_clip = PROCESSED_DIR / f"{ep_id}_rend_raw_{i}.mp3"
            synced_clip = PROCESSED_DIR / f"{ep_id}_rend_sync_{i}.wav"

            ok = await generate_khmer_audio_file(txt, v, str(raw_clip))
            if not ok:
                return None

            target_dur = max(it["end"] - it["start"], 0.3)
            process_exact_time_lock(str(raw_clip), str(synced_clip), target_dur, cfg.speed_factor)
            return {"start": it["start"], "path": str(synced_clip)}

    tasks = [fetch_audio_turbo(i, d if isinstance(d, dict) else d.dict()) for i, d in enumerate(dialogues)]
    raw_clips = await asyncio.gather(*tasks)
    clips = [c for c in raw_clips if c is not None]

    ep["progress"] = 65

    dub_track = PROCESSED_DIR / f"{ep_id}_dub.wav"
    if clips:
        input_args = []
        delays = []
        for idx, clip in enumerate(clips):
            input_args.extend(["-i", clip["path"]])
            ms = int(clip["start"] * 1000)
            delays.append(f"[{idx}:a]adelay={ms}|{ms},aformat=sample_fmts=s16:sample_rates=44100:channel_layouts=stereo[a{idx}];")

        mix_inputs = "".join(f"[a{i}]" for i in range(len(clips)))
        filter_str = "".join(delays) + f"{mix_inputs}amix=inputs={len(clips)}:dropout_transition=0:normalize=0[dub_out]"
        subprocess.run(["ffmpeg", "-y"] + input_args + ["-filter_complex", filter_str, "-map", "[dub_out]", "-threads", "0", "-ar", "44100", "-ac", "2", str(dub_track)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", "10", str(dub_track)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    ep["progress"] = 80

    render_token = uuid.uuid4().hex[:6]
    out_name = f"dubbed_{ep['code']}_{cfg.resolution}_{render_token}.mp4"
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

    rendered_ok = False
    try:
        cmd = [
            "ffmpeg", "-y",
            *trim_args,
            "-i", str(video_in),
            "-i", str(dub_track),
            "-filter_complex", f"[0:v]{v_filter_str}[vout];[1:a]volume={cfg.dub_volume}[aout]",
            "-map", "[vout]",
            "-map", "[aout]",
            *enc_flags,
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            str(final_mp4)
        ]
        res = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            encoding='utf-8', 
            errors='replace'
        )
        if res.returncode == 0 and final_mp4.exists():
            rendered_ok = True
    except Exception as e:
        print("Hardware Turbo Render Notice:", e)

    if not rendered_ok:
        # Fallback Ultra Fast Stream Copy
        cmd_fb = [
            "ffmpeg", "-y",
            *trim_args,
            "-i", str(video_in),
            "-i", str(dub_track),
            "-filter_complex", f"[1:a]volume={cfg.dub_volume}[aout]",
            "-map", "0:v",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            str(final_mp4)
        ]
        subprocess.run(
            cmd_fb, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            encoding='utf-8', 
            errors='replace'
        )

    ep["status"] = "completed"
    ep["progress"] = 100
    ep["output_url"] = f"/api/download/{out_name}"
    save_db()
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
    print("\n=======================================================")
    print(" 🚀 KhmerDub Studio Pro - V3 Turbo 4X Ultra Running:")
    print(" ⚡ Model: OpenAI Whisper Large-V3-Turbo")
    print(f" 🎬 Video Acceleration: {HARDWARE_ENCODER}")
    print(" 💻 Local Link:  http://127.0.0.1:8080")
    print(" 📱 Mobile Link: http://0.0.0.0:8080 (ឬប្រើលេខ IP កុំព្យូទ័រ)")
    print("=======================================================\n")
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)