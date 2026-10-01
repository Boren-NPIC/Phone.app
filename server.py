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

# រក FFmpeg ស្វ័យប្រវត្តិតាម imageio_ffmpeg បើម៉ាស៊ីនគ្មាន ffmpeg ក្នុង PATH
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

app = FastAPI(title="KhmerDub Studio Pro - Cloud Optimized", version="340.0.0")

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
    if DB_FILE.exists():
        try:
            with open(DB_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    EPISODES_DB = data
        except Exception:
            pass

def save_db():
    try:
        with open(DB_FILE, "w", encoding="utf-8") as f:
            json.dump(EPISODES_DB, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[!] Save DB Error: {e}")

load_db()

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
            FFMPEG_BIN, "-i", str(file_path)
        ]
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

def get_whisper_turbo_model(model_name: str = "tiny"):
    global GLOBAL_WHISPER_INSTANCE, CURRENT_MODEL_NAME
    # បើដំណើរការលើ Render (RAM 512MB) បង្ខំឱ្យប្រើ tiny ដើម្បីកុំឱ្យ Crash 502
    is_render = bool(os.environ.get("RENDER"))
    chosen_model = "tiny" if is_render else model_name

    if GLOBAL_WHISPER_INSTANCE is None or CURRENT_MODEL_NAME != chosen_model:
        if WhisperModel:
            try:
                GLOBAL_WHISPER_INSTANCE = WhisperModel(
                    chosen_model, 
                    device="cpu", 
                    compute_type="int8", 
                    cpu_threads=2,
                    num_workers=1
                )
                CURRENT_MODEL_NAME = chosen_model
            except Exception as e:
                print(f"[!] Whisper Load Error: {e}")
                GLOBAL_WHISPER_INSTANCE = WhisperModel("tiny", device="cpu", compute_type="int8")
                CURRENT_MODEL_NAME = "tiny"
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
        "1. Match conversational emotion and scene context.\n"
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
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {cloud_key.strip()}"
                }
            )
            with urllib.request.urlopen(req, timeout=12) as resp:
                res_json = json.loads(resp.read().decode("utf-8"))
                content = res_json["choices"][0]["message"]["content"].strip()
                match = re.search(r'\[.*\]', content, re.DOTALL)
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
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {cloud_key.strip()}"
                }
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                res_json = json.loads(resp.read().decode("utf-8"))
                content = res_json["choices"][0]["message"]["content"].strip()
                match = re.search(r'\[.*\]', content, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                    if len(parsed) == len(texts):
                        return [str(p) for p in parsed]
        except Exception:
            pass

    results = []
    src_code = "auto" if source_lang == "auto" else ("zh-CN" if source_lang == "zh" else "en")

    for txt in texts:
        clean = txt.strip()
        if not clean:
            results.append("")
            continue
        clean = re.sub(r'^[（\(].*?[）\)]', '', clean)
        
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
            pass

        if not translated:
            translated = clean

        cleaned_km = translated.replace("នារីម្នាក់", "នាង").replace("បុរសម្នាក់", "គាត់")
        results.append(cleaned_km)

    return results

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

@app.post("/api/transcribe-episode")
async def transcribe_episode(
    episode_id: str = Form(...),
    model_size: str = Form("tiny"),
    cloud_key: str = Form(""),
    lang: str = Form("auto"),
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

        # បំបែកសំឡេងតាម FFMPEG_BIN ធានាថាមិន Error 502
        subprocess.run([
            FFMPEG_BIN, "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(audio_path)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        whisper = get_whisper_turbo_model("tiny")
        sub_segments = []

        if whisper:
            whisper_lang = None if lang == "auto" else lang
            segments, _ = whisper.transcribe(
                str(audio_path),
                vad_filter=True,
                language=whisper_lang,
                beam_size=1,
                temperature=0.0
            )

            for seg in segments:
                txt = seg.text.strip()
                if txt:
                    sub_segments.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "text": txt})

        orig_texts = [s["text"] for s in sub_segments]
        khmer_translations = translate_batch_with_ai_cloud(orig_texts, source_lang=lang, cloud_type=model_size, cloud_key=cloud_key)

        dialogues = []
        last_speaker = "piseth"

        for idx, orig in enumerate(sub_segments):
            final_khmer = khmer_translations[idx] if idx < len(khmer_translations) else orig["text"]
            final_spk = "piseth" if speaker_mode == "male" else ("sreymom" if speaker_mode == "female" else analyze_smart_speaker(orig["text"], final_khmer, last_speaker))
            last_speaker = final_spk
            dialogues.append({
                "id": idx + 1,
                "start": orig["start"],
                "end": orig["end"],
                "original_text": orig["text"],
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
async def preview_single_tts(text: str = Form(...), speaker: str = Form("piseth")):
    clean_text = text.strip() or "សួស្តី"
    voice = "km-KH-SreymomNeural" if speaker == "sreymom" else "km-KH-PisethNeural"
    uid = uuid.uuid4().hex[:6]
    out_wav = PROCESSED_DIR / f"prev_{uid}.wav"

    ok = generate_khmer_audio_fast(clean_text, voice, str(out_wav))
    if not ok or not out_wav.exists():
        raise HTTPException(status_code=500, detail="TTS Engine Failed")

    return FileResponse(path=str(out_wav), media_type="audio/wav", filename="preview.wav")

def render_single_episode_sync(ep_id: str, cfg_dict: dict) -> str:
    load_db()
    ep = EPISODES_DB[ep_id]
    ep["status"] = "rendering"
    ep["progress"] = 30
    save_db()

    dialogues = cfg_dict.get("dialogues_map", {}).get(ep_id) or ep.get("dialogues", [])
    video_in = ep["path"]
    total_video_duration = get_media_duration(str(video_in)) or 60.0

    clips_results = []
    for idx, d in enumerate(dialogues):
        txt = d.get("khmer_text", "").strip()
        if not txt:
            continue
        voice_id = "km-KH-SreymomNeural" if d.get("speaker") == "sreymom" else "km-KH-PisethNeural"
        raw_wav = PROCESSED_DIR / f"{ep_id}_r_{idx}.wav"
        fit_wav = PROCESSED_DIR / f"{ep_id}_f_{idx}.wav"

        if generate_khmer_audio_fast(txt, voice_id, str(raw_wav)):
            fit_audio_exact_to_scene(str(raw_wav), max(0.35, float(d["end"]) - float(d["start"])), str(fit_wav))
            clips_results.append({
                "start": float(d["start"]),
                "duration": get_media_duration(str(fit_wav)),
                "path": str(fit_wav)
            })

    ep["progress"] = 70
    save_db()

    dub_track = PROCESSED_DIR / f"{ep_id}_dub.wav"
    manifest_txt = PROCESSED_DIR / f"{ep_id}_manifest.txt"
    current_cursor = 0.0

    common_silence = PROCESSED_DIR / "silence_master.wav"
    if not common_silence.exists():
        subprocess.run([
            FFMPEG_BIN, "-y", "-f", "lavfi",
            "-i", "anullsrc=r=48000:cl=stereo",
            "-t", "30",
            "-c:a", "pcm_s16le",
            str(common_silence)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    with open(manifest_txt, "w", encoding="utf-8") as f_mf:
        for idx, clip in enumerate(clips_results):
            gap = clip["start"] - current_cursor
            if gap > 0.04:
                sil_chunk = PROCESSED_DIR / f"{ep_id}_sil_{idx}.wav"
                subprocess.run([FFMPEG_BIN, "-y", "-i", str(common_silence), "-t", f"{gap:.3f}", "-c", "copy", str(sil_chunk)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                f_mf.write(f"file '{str(sil_chunk).replace(os.sep, '/')}'\n")

            f_mf.write(f"file '{clip['path'].replace(os.sep, '/')}'\n")
            current_cursor = clip["start"] + clip["duration"]

        if total_video_duration > current_cursor:
            end_sil = PROCESSED_DIR / f"{ep_id}_end_sil.wav"
            subprocess.run([FFMPEG_BIN, "-y", "-i", str(common_silence), "-t", f"{total_video_duration - current_cursor:.3f}", "-c", "copy", str(end_sil)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            f_mf.write(f"file '{str(end_sil).replace(os.sep, '/')}'\n")

    subprocess.run([
        FFMPEG_BIN, "-y", "-f", "concat", "-safe", "0",
        "-i", str(manifest_txt),
        "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2",
        str(dub_track)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    ep["progress"] = 90
    save_db()

    out_name = f"dubbed_{ep.get('code', 'ep')}_{uuid.uuid4().hex[:6]}.mp4"
    final_mp4 = PROCESSED_DIR / out_name

    cmd = [
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
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    load_db()
    if ep_id in EPISODES_DB:
        EPISODES_DB[ep_id]["status"] = "completed"
        EPISODES_DB[ep_id]["progress"] = 100
        EPISODES_DB[ep_id]["output_url"] = f"/api/download/{out_name}"
        save_db()

    return str(final_mp4)

def run_render_in_daemon_thread(cfg_dict: dict):
    ep_ids = cfg_dict.get("episode_ids", [])
    for ep_id in ep_ids:
        load_db()
        if ep_id in EPISODES_DB:
            try:
                render_single_episode_sync(ep_id, cfg_dict)
            except Exception as e:
                print(f"[!] Render error: {e}")
                EPISODES_DB[ep_id]["status"] = "failed"
                save_db()

@app.post("/api/batch-render")
def start_batch_render(cfg: BatchRenderConfig):
    worker = threading.Thread(target=run_render_in_daemon_thread, args=(cfg.dict(),), daemon=True)
    worker.start()
    return JSONResponse({"message": "Batch rendering started"})

@app.get("/api/episodes")
def get_episodes():
    load_db()
    return JSONResponse(EPISODES_DB)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)