from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from functools import lru_cache
import tempfile
import os
import uvicorn
from transformers import pipeline

# --- FastAPI app init ---
app = FastAPI()

# --- Allow CORS for frontend ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Lightweight model pipelines (optimized for Render Free Tier) ---
@lru_cache()
def get_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")  # ~300MB RAM

@lru_cache()
def get_asr():
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny")  # ~200MB RAM

# --- In-memory buffer for live transcribe ---
buffered_transcripts = []

# --- Pydantic schema for assignment evaluation ---
class AssignmentRequest(BaseModel):
    criteria: str
    brief: str
    assignment: str

@app.post("/evaluate")
async def evaluate(data: AssignmentRequest):
    combined = f"Criteria:\n{data.criteria}\n\nBrief:\n{data.brief}\n\nAssignment:\n{data.assignment}"
    if len(combined) > 1024:
        combined = combined[:1024]

    try:
        summarizer = get_summarizer()
        result = summarizer(combined, max_length=120, min_length=50, do_sample=False)
        return {"feedback": result[0]["summary_text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

@app.post("/process_audio")
async def process_audio(audio: UploadFile = File(...)):
    _, ext = os.path.splitext(audio.filename)
    if not ext:
        ext = ".wav"

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(await audio.read())
        temp_path = tmp.name

    try:
        asr = get_asr()
        transcription = asr(temp_path)["text"].strip()

        wc = len(transcription.split())
        min_len = 10 if wc < 60 else 50
        max_len = 50 if wc < 120 else 120

        summarizer = get_summarizer()
        summary = summarizer(transcription, max_length=max_len, min_length=min_len, do_sample=False)[0]["summary_text"]

        return {"transcription": transcription, "summary": summary}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ASR or summarization failed: {str(e)}")

    finally:
        os.remove(temp_path)

@app.post("/live_transcribe")
async def live_transcribe(audio: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(await audio.read())
        temp_path = tmp.name

    try:
        text = get_asr()(temp_path)["text"].strip()
        buffered_transcripts.append(text)
        return {"transcript": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Live transcription failed: {str(e)}")
    finally:
        os.remove(temp_path)

@app.get("/get_summary")
async def get_summary():
    full_text = " ".join(buffered_transcripts)
    buffered_transcripts.clear()
    if not full_text:
        return {"summary": ""}

    try:
        summarizer = get_summarizer()
        result = summarizer(full_text, max_length=120, min_length=50, do_sample=False)
        return {"summary": result[0]["summary_text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summary failed: {str(e)}")

@app.get("/")
def root():
    return {"message": "FastAPI server is up and running 🚀"}

if __name__ == "__main__":
    uvicorn.run("servertest:app", host="0.0.0.0", port=8000, reload=True)
