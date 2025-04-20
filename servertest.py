from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
import tempfile
import os
import uvicorn

# Init FastAPI app
app = FastAPI()

# CORS config (cho phép gọi từ frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load AI pipelines
assignment_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-base")

# Buffer để lưu transcript từ live recording
buffered_transcripts = []

class AssignmentRequest(BaseModel):
    criteria: str
    brief: str
    assignment: str

@app.post("/evaluate")
async def evaluate(data: AssignmentRequest):
    combined = f"Criteria:\n{data.criteria}\n\nBrief:\n{data.brief}\n\nAssignment:\n{data.assignment}"
    if len(combined) > 1024:
        combined = combined[:1024]

    result = assignment_summarizer(
        combined,
        max_length=150,
        min_length=60,
        do_sample=False
    )
    return {"feedback": result[0]["summary_text"]}

@app.post("/process_audio")
async def process_audio(audio: UploadFile = File(...)):
    _, ext = os.path.splitext(audio.filename)
    if not ext:
        ext = ".wav"

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(await audio.read())
        temp_path = tmp.name

    try:
        transcription = asr_pipeline(temp_path)["text"].strip()
        wc = len(transcription.split())
        min_len = 10 if wc < 60 else 60
        max_len = 60 if wc < 120 else 150

        summary = assignment_summarizer(
            transcription,
            max_length=max_len,
            min_length=min_len,
            do_sample=False
        )[0]["summary_text"]

        return {
            "transcription": transcription,
            "summary": summary
        }
    finally:
        os.remove(temp_path)

@app.post("/live_transcribe")
async def live_transcribe(audio: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(await audio.read())
        temp_path = tmp.name

    try:
        text = asr_pipeline(temp_path)["text"].strip()
        buffered_transcripts.append(text)
        return {"transcript": text}
    finally:
        os.remove(temp_path)

@app.get("/get_summary")
async def get_summary():
    full_text = " ".join(buffered_transcripts)
    buffered_transcripts.clear()
    if not full_text:
        return {"summary": ""}

    result = assignment_summarizer(
        full_text,
        max_length=150,
        min_length=60,
        do_sample=False
    )
    return {"summary": result[0]["summary_text"]}
@app.get("/")
def root():
    return {"message": "FastAPI server is up and running 🚀"}
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
#uvicorn servertest:app --reload