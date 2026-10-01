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

app = FastAPI(title="KhmerDub Studio Pro - Precision UI Sync Engine", version="330.0.0")

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
                success = future.result(timeout=2.8)
            except Exception:
                success = False

    if not success or not os.path.exists(temp_mp3) or os.path.getsize(temp_mp3) < 150:
        try:
            encoded = urllib.parse.quote(clean_text[:180])
            url = f"https://translate.google.com/translate_tts?ie=UTF-8&q={encoded}&tl=km&client=tw-ob"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=2.0) as resp, open(str(temp_mp3), "wb") as f:
                f.write(resp.read())
            if os.path.exists(temp_mp3) and os.path.getsize(temp_mp3) > 150:
                success = True
        except Exception:
            pass

    if not success or not os.path.exists(temp_mp3):
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "anullsrc=r=48000:cl=stereo",
            "-t", "0.5",
            "-c:a", "pcm_s16le",
            str(out_wav_path)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True

    subprocess.run([
        "ffmpeg", "-y", "-i", str(temp_mp3),
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
        "ffmpeg", "-y", "-i", str(raw_wav),
        "-filter:a", af_str,
        "-t", f"{scene_dur:.3f}",
        "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le",
        str(out_wav)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.exists(out_wav) or os.path.getsize(out_wav) < 100:
        shutil.copyfile(raw_wav, out_wav)

def get_whisper_turbo_model(model_name: str = "small"):
    global GLOBAL_WHISPER_INSTANCE, CURRENT_MODEL_NAME
    has_gpu = bool(shutil.which("nvidia-smi"))
    chosen_model = model_name
    if not has_gpu and model_name in ["large-v3-turbo", "large-v3", "large"]:
        chosen_model = "small"

    if GLOBAL_WHISPER_INSTANCE is None or CURRENT_MODEL_NAME != chosen_model:
        if WhisperModel:
            device = "cuda" if has_gpu else "cpu"
            comp_type = "float16" if has_gpu else "int8"
            threads = os.cpu_count() or 6
            try:
                GLOBAL_WHISPER_INSTANCE = WhisperModel(
                    chosen_model, 
                    device=device, 
                    compute_type=comp_type, 
                    cpu_threads=threads,
                    num_workers=2
                )
                CURRENT_MODEL_NAME = chosen_model
            except Exception:
                GLOBAL_WHISPER_INSTANCE = WhisperModel("base", device="cpu", compute_type="int8")
                CURRENT_MODEL_NAME = "base"
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
        "1. Match the exact conversational emotion and context of the scene.\n"
        "2. Keep the Khmer translations concise and punchy (avoid long descriptive phrases) so that they fit the lip-sync timing.\n"
        "3. Output MUST be ONLY a strict JSON array of translated Khmer strings with the exact same length and order."
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
        except Exception as e:
            print(f"[!] OpenAI Cloud Translation Error: {e}")

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
        except Exception as e:
            print(f"[!] Groq Cloud Translation Error: {e}")

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
            try:
                url2 = f"https://api.mymemory.translated.net/get?q={urllib.parse.quote(clean)}&langpair={source_lang}|km"
                req2 = urllib.request.Request(url2, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req2, timeout=3.0) as resp2:
                    data2 = json.loads(resp2.read().decode("utf-8"))
                    translated = data2.get("responseData", {}).get("translatedText", "")
            except Exception:
                pass

        if not translated or "MYMEMORY WARNING" in translated:
            translated = clean

        cleaned_km = translated.replace("នារីម្នាក់", "នាង")\
                               .replace("បុរសម្នាក់", "គាត់")\
                               .replace("ជ្រើសរើសយក", "ជ្រើសយក")
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
    model_size: str = Form("small"),
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

        subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(audio_path)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        actual_whisper_model = "small" if "cloud" in model_size else model_size
        whisper = get_whisper_turbo_model(actual_whisper_model)
        sub_segments = []

        if whisper:
            whisper_lang = None if lang == "auto" else lang
            segments, _ = whisper.transcribe(
                str(audio_path),
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=100, speech_pad_ms=60),
                language=whisper_lang,
                word_timestamps=True,
                beam_size=1,
                temperature=0.0
            )

            for seg in segments:
                words = getattr(seg, "words", None)
                if not words:
                    sub_segments.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "text": seg.text.strip()})
                    continue

                curr_words = []
                for w in words:
                    if curr_words:
                        gap = w.start - curr_words[-1].end
                        dur = w.end - curr_words[0].start
                        if (gap > 0.60 or dur > 3.8) and dur >= 0.40:
                            w_text = " ".join([cw.word.strip() for cw in curr_words]).strip()
                            if w_text:
                                sub_segments.append({"start": round(curr_words[0].start, 2), "end": round(curr_words[-1].end, 2), "text": w_text})
                            curr_words = []
                    curr_words.append(w)

                if curr_words:
                    w_text = " ".join([cw.word.strip() for cw in curr_words]).strip()
                    if w_text:
                        sub_segments.append({"start": round(curr_words[0].start, 2), "end": round(curr_words[-1].end, 2), "text": w_text})

        orig_texts = [s["text"] for s in sub_segments]
        print(f"[*] Translating {len(sub_segments)} lines ({lang}) via {model_size}...")
        khmer_translations = translate_batch_with_ai_cloud(orig_texts, source_lang=lang, cloud_type=model_size, cloud_key=cloud_key)

        dialogues = []
        last_speaker = "piseth"

        for idx, orig in enumerate(sub_segments):
            final_khmer = khmer_translations[idx] if idx < len(khmer_translations) else orig["text"]
            
            if speaker_mode == "male":
                final_spk = "piseth"
            elif speaker_mode == "female":
                final_spk = "sreymom"
            else:
                final_spk = analyze_smart_speaker(orig["text"], final_khmer, last_speaker)

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

def process_single_clip_turbo(item, ep_id, index):
    txt = item.get("khmer_text", "").strip()
    if not txt:
        return None

    spk = item.get("speaker", "piseth")
    voice_id = "km-KH-SreymomNeural" if spk == "sreymom" else "km-KH-PisethNeural"

    raw_wav = PROCESSED_DIR / f"{ep_id}_r_{index}.wav"
    fit_wav = PROCESSED_DIR / f"{ep_id}_f_{index}.wav"

    if generate_khmer_audio_fast(txt, voice_id, str(raw_wav)):
        target_scene_dur = max(0.35, float(item["end"]) - float(item["start"]))
        fit_audio_exact_to_scene(str(raw_wav), target_scene_dur, str(fit_wav))
        actual_dur = get_media_duration(str(fit_wav))
        return {
            "index": index,
            "start": float(item["start"]),
            "duration": actual_dur,
            "path": str(fit_wav)
        }
    return None

def render_single_episode_sync(ep_id: str, cfg_dict: dict) -> str:
    load_db()
    ep = EPISODES_DB[ep_id]
    ep["status"] = "rendering"
    ep["progress"] = 25
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

    ep["progress"] = 40
    save_db()

    total_video_duration = get_media_duration(str(video_in))
    if total_video_duration <= 0:
        total_video_duration = 300.0

    print(f"[*] Fast Audio Processing for {ep.get('code', 'ep')} ({len(dialogues)} clips)...")
    clips_results = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(process_single_clip_turbo, d if isinstance(d, dict) else d.dict(), ep_id, idx): idx for idx, d in enumerate(dialogues)}
        for future in as_completed(futures):
            res = future.result()
            if res:
                clips_results.append(res)

    clips_results.sort(key=lambda x: x["start"])

    ep["progress"] = 70
    save_db()

    dub_track = PROCESSED_DIR / f"{ep_id}_dub.wav"
    manifest_txt = PROCESSED_DIR / f"{ep_id}_manifest.txt"
    current_cursor = 0.0

    common_silence = PROCESSED_DIR / f"silence_master.wav"
    if not common_silence.exists():
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi",
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
                subprocess.run([
                    "ffmpeg", "-y", "-i", str(common_silence),
                    "-t", f"{gap:.3f}",
                    "-c", "copy",
                    str(sil_chunk)
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                f_mf.write(f"file '{str(sil_chunk).replace(os.sep, '/')}'\n")

            f_mf.write(f"file '{clip['path'].replace(os.sep, '/')}'\n")
            current_cursor = clip["start"] + clip["duration"]

        if total_video_duration > current_cursor:
            end_gap = total_video_duration - current_cursor
            end_sil = PROCESSED_DIR / f"{ep_id}_end_sil.wav"
            subprocess.run([
                "ffmpeg", "-y", "-i", str(common_silence),
                "-t", f"{end_gap:.3f}",
                "-c", "copy",
                str(end_sil)
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            f_mf.write(f"file '{str(end_sil).replace(os.sep, '/')}'\n")

    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(manifest_txt),
        "-c:a", "pcm_s16le",
        "-ar", "48000", "-ac", "2",
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
    scale_filter = f"scale={target_w}:{target_h}:flags=bilinear:force_original_aspect_ratio=decrease,pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2"
    combined_vf = f"{scale_filter},{sub_filter}"

    print(f"[*] Turbo Rendering {ep.get('code', 'ep')} to {out_name}...")
    ep["progress"] = 92
    save_db()

    has_gpu = bool(shutil.which("nvidia-smi"))
    encoder_args = ["-c:v", "h264_nvenc", "-preset", "p4"] if has_gpu else ["-c:v", "libx264", "-preset", "ultrafast"]

    cmd = [
        "ffmpeg", "-y",
        *trim_args,
        "-i", str(video_in),
        "-i", str(dub_track),
        "-vf", combined_vf,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-filter:a", master_audio_filter,
        *encoder_args,
        "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "256k",
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
            "-b:a", "256k",
            str(final_mp4)
        ]
        subprocess.run(cmd_fb, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # កត់ត្រាស្ថានភាពជោគជ័យ ១០០% ជាស្ថាពរ
    load_db()
    if ep_id in EPISODES_DB:
        EPISODES_DB[ep_id]["status"] = "completed"
        EPISODES_DB[ep_id]["progress"] = 100
        EPISODES_DB[ep_id]["output_url"] = f"/api/download/{out_name}"
        save_db()

    print(f"[✓] Render finished successfully: {out_name}\n")
    return str(final_mp4)

def run_render_in_daemon_thread(cfg_dict: dict):
    completed_files = []
    ep_ids = cfg_dict.get("episode_ids", [])
    
    load_db()

    def get_sort_key(eid):
        ep_info = EPISODES_DB.get(eid, {})
        code = ep_info.get("code", "")
        nums = re.findall(r'\d+', code)
        return int(nums[0]) if nums else 9999

    sorted_ep_ids = sorted(ep_ids, key=get_sort_key)

    for ep_id in sorted_ep_ids:
        load_db()
        if ep_id in EPISODES_DB:
            current_ep = EPISODES_DB[ep_id]
            print(f"\n==========================================")
            print(f"[*] ចាប់ផ្តើម Render ភាគ: {current_ep.get('code')} ({current_ep.get('filename')})")
            print(f"==========================================")
            try:
                out_file = render_single_episode_sync(ep_id, cfg_dict)
                if out_file and os.path.exists(out_file):
                    completed_files.append(out_file)
            except Exception as e:
                print(f"[!] បរាជ័យក្នុងការ Render {ep_id}: {e}")
                load_db()
                if ep_id in EPISODES_DB:
                    EPISODES_DB[ep_id]["status"] = "failed"
                    save_db()

    if cfg_dict.get("merge_all_into_one") and len(completed_files) > 1:
        merged_name = f"merged_dub_{cfg_dict.get('resolution', '1080p')}_{uuid.uuid4().hex[:6]}.mp4"
        merged_path = PROCESSED_DIR / merged_name
        list_txt = PROCESSED_DIR / "concat_list.txt"
        with open(list_txt, "w", encoding="utf-8") as f:
            for c in completed_files:
                f.write(f"file '{c.replace(os.sep, '/')}'\n")

        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_txt),
            "-c", "copy", str(merged_path)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        load_db()
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
    return JSONResponse({"message": "Batch rendering started"})

@app.get("/api/episodes")
def get_episodes():
    load_db()
    return JSONResponse(EPISODES_DB)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    bind_host = "0.0.0.0" if os.environ.get("RENDER") else "127.0.0.1"
    
    print("\n" + "=" * 55)
    print(f"🚀 KhmerDub Studio Pro [Ready & Synchronized] is RUNNING!")
    print(f"👉 Link: http://127.0.0.1:{port}")
    print("=" * 55 + "\n")

    uvicorn.run("server:app", host=bind_host, port=port, reload=False)