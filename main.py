from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import asyncio
import json
import base64
import os
from typing import Optional
import logging
from pydantic import BaseModel
import google.generativeai as genai
import requests
import time
from datetime import datetime
from dotenv import load_dotenv
import pathlib
import aiofiles

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sehatnama - Clinical Notes Generator", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = pathlib.Path("static")
static_dir.mkdir(exist_ok=True)

temp_dir = pathlib.Path("temp")
temp_dir.mkdir(exist_ok=True)

index_file = static_dir / "index.html"
if not index_file.exists():
    logger.warning("index.html not found in static directory")

app.mount("/static", StaticFiles(directory="static"), name="static")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")
if not ASSEMBLYAI_API_KEY:
    raise ValueError("ASSEMBLYAI_API_KEY environment variable is required")

genai.configure(api_key=GEMINI_API_KEY)

ASSEMBLYAI_BASE_URL = "https://api.assemblyai.com"
ASSEMBLYAI_HEADERS = {
    "authorization": ASSEMBLYAI_API_KEY
}

class NoteRequest(BaseModel):
    transcript: str
    note_type: str = "SOAP"
    custom_prompt: Optional[str] = None

class AudioRecordingRequest(BaseModel):
    audio_data: str
    note_type: str = "SOAP"

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            self.disconnect(websocket)

manager = ConnectionManager()

SOAP_TEMPLATE = """
You are a medical professional assistant. Based on the following transcript of a patient encounter, 
generate a structured SOAP note. Be thorough, accurate, and maintain professional medical terminology.

SOAP Format:
- S (Subjective): Patient's reported symptoms, concerns, and medical history
- O (Objective): Observable findings, vital signs, physical examination results
- A (Assessment): Medical assessment, diagnosis, or clinical impression
- P (Plan): Treatment plan, medications, follow-up instructions

Transcript: {transcript}

Generate a well-structured SOAP note:
"""

BIRP_TEMPLATE = """
You are a mental health professional assistant. Based on the following transcript of a therapy/counseling session,
generate a structured BIRP note. Be professional, objective, and maintain client confidentiality standards.

BIRP Format:
- B (Behavior): Observable behaviors, presentation, and engagement
- I (Intervention): Therapeutic interventions used during the session
- R (Response): Client's response to interventions and progress
- P (Plan): Treatment plan, goals, and next session planning

Transcript: {transcript}

Generate a well-structured BIRP note:
"""

async def generate_clinical_note(transcript: str, note_type: str = "SOAP", custom_prompt: str = None) -> str:
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-lite')
        
        if custom_prompt:
            prompt = custom_prompt.format(transcript=transcript)
        elif note_type.upper() == "BIRP":
            prompt = BIRP_TEMPLATE.format(transcript=transcript)
        else:
            prompt = SOAP_TEMPLATE.format(transcript=transcript)
        
        response = await asyncio.to_thread(model.generate_content, prompt)
        
        return response.text
    except Exception as e:
        logger.error(f"Error generating clinical note: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate clinical note: {str(e)}")

async def upload_audio_to_assemblyai(file_path: str) -> str:
    try:
        async with aiofiles.open(file_path, 'rb') as f:
            audio_data = await f.read()
        
        upload_response = await asyncio.to_thread(
            requests.post,
            f"{ASSEMBLYAI_BASE_URL}/v2/upload",
            headers=ASSEMBLYAI_HEADERS,
            data=audio_data
        )
        
        if upload_response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to upload audio to AssemblyAI")
        
        return upload_response.json()["upload_url"]
    except Exception as e:
        logger.error(f"Error uploading audio to AssemblyAI: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload audio: {str(e)}")

async def transcribe_audio_with_assemblyai(audio_url: str) -> str:
    try:
        data = {
            "audio_url": audio_url,
            "speech_model": "universal"
        }
        
        response = await asyncio.to_thread(
            requests.post,
            f"{ASSEMBLYAI_BASE_URL}/v2/transcript",
            json=data,
            headers=ASSEMBLYAI_HEADERS
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to submit transcription request")
        
        transcript_id = response.json()['id']
        polling_endpoint = f"{ASSEMBLYAI_BASE_URL}/v2/transcript/{transcript_id}"
        
        while True:
            result = await asyncio.to_thread(
                requests.get,
                polling_endpoint,
                headers=ASSEMBLYAI_HEADERS
            )
            
            if result.status_code != 200:
                raise HTTPException(status_code=500, detail="Failed to get transcription status")
            
            transcription_result = result.json()
            
            if transcription_result['status'] == 'completed':
                return transcription_result['text']
            elif transcription_result['status'] == 'error':
                raise HTTPException(
                    status_code=500, 
                    detail=f"Transcription failed: {transcription_result.get('error', 'Unknown error')}"
                )
            else:
                await asyncio.sleep(3)
                
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to transcribe audio: {str(e)}")

@app.get("/", response_class=HTMLResponse)
@app.head("/")
async def get_index():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html>
            <body>
                <h1>Sehatnama - Clinical Notes Generator</h1>
                <p>Frontend file not found. Please add index.html to the static directory.</p>
                <p>Available API endpoints:</p>
                <ul>
                    <li>POST /generate-note - Generate note from transcript</li>
                    <li>POST /upload-audio - Upload and process audio file</li>
                    <li>POST /record-audio - Process recorded audio</li>
                </ul>
            </body>
        </html>
        """)
async def get_index():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html>
            <body>
                <h1>Sehatnama - Clinical Notes Generator</h1>
                <p>Frontend file not found. Please add index.html to the static directory.</p>
                <p>Available API endpoints:</p>
                <ul>
                    <li>POST /generate-note - Generate note from transcript</li>
                    <li>POST /upload-audio - Upload and process audio file</li>
                    <li>POST /record-audio - Process recorded audio</li>
                </ul>
            </body>
        </html>
        """)

@app.post("/generate-note")
async def generate_note_endpoint(request: NoteRequest):
    try:
        note = await generate_clinical_note(
            request.transcript, 
            request.note_type, 
            request.custom_prompt
        )
        
        return {
            "success": True,
            "note": note,
            "note_type": request.note_type,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in generate_note_endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-audio")
async def upload_audio_file(file: UploadFile = File(...), note_type: str = "SOAP"):
    temp_file_path = None
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_file_path = f"temp/audio_{timestamp}_{file.filename}"
        
        async with aiofiles.open(temp_file_path, "wb") as buffer:
            content = await file.read()
            await buffer.write(content)
        
        audio_url = await upload_audio_to_assemblyai(temp_file_path)
        transcript_text = await transcribe_audio_with_assemblyai(audio_url)
        note = await generate_clinical_note(transcript_text, note_type)
        
        return {
            "success": True,
            "transcript": transcript_text,
            "note": note,
            "note_type": note_type,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing audio file: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                logger.error(f"Error cleaning up temp file: {e}")

@app.post("/record-audio")
async def process_recorded_audio(request: AudioRecordingRequest):
    temp_file_path = None
    try:
        audio_data = base64.b64decode(request.audio_data)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_file_path = f"temp/recorded_{timestamp}.wav"
        
        async with aiofiles.open(temp_file_path, "wb") as f:
            await f.write(audio_data)
        
        audio_url = await upload_audio_to_assemblyai(temp_file_path)
        transcript_text = await transcribe_audio_with_assemblyai(audio_url)
        note = await generate_clinical_note(transcript_text, request.note_type)
        
        return {
            "success": True,
            "transcript": transcript_text,
            "note": note,
            "note_type": request.note_type,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing recorded audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                logger.error(f"Error cleaning up temp file: {e}")

@app.websocket("/ws/audio-recording")
async def websocket_audio_recording(websocket: WebSocket):
    await manager.connect(websocket)
    
    try:
        await manager.send_personal_message(
            json.dumps({"type": "connected", "message": "Recording session started"}),
            websocket
        )
        
        while True:
            data = await websocket.receive()
            
            if "text" in data:
                message = json.loads(data["text"])
                
                if message["type"] == "audio_complete":
                    audio_data = message["audio_data"]
                    note_type = message.get("note_type", "SOAP")
                    
                    try:
                        await manager.send_personal_message(
                            json.dumps({"type": "status", "message": "Processing audio..."}),
                            websocket
                        )
                        
                        audio_bytes = base64.b64decode(audio_data)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        temp_file_path = f"temp/ws_recorded_{timestamp}.wav"
                        
                        async with aiofiles.open(temp_file_path, "wb") as f:
                            await f.write(audio_bytes)
                        
                        audio_url = await upload_audio_to_assemblyai(temp_file_path)
                        
                        await manager.send_personal_message(
                            json.dumps({"type": "status", "message": "Transcribing audio..."}),
                            websocket
                        )
                        
                        transcript_text = await transcribe_audio_with_assemblyai(audio_url)
                        
                        await manager.send_personal_message(
                            json.dumps({"type": "transcript", "text": transcript_text}),
                            websocket
                        )
                        
                        await manager.send_personal_message(
                            json.dumps({"type": "status", "message": "Generating clinical note..."}),
                            websocket
                        )
                        
                        note = await generate_clinical_note(transcript_text, note_type)
                        
                        await manager.send_personal_message(
                            json.dumps({
                                "type": "clinical_note",
                                "transcript": transcript_text,
                                "note": note,
                                "note_type": note_type
                            }),
                            websocket
                        )
                        
                        if os.path.exists(temp_file_path):
                            os.remove(temp_file_path)
                            
                    except Exception as e:
                        logger.error(f"Error processing WebSocket audio: {e}")
                        await manager.send_personal_message(
                            json.dumps({"type": "error", "message": str(e)}),
                            websocket
                        )
                    
                    break
                    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await manager.send_personal_message(
            json.dumps({"type": "error", "message": str(e)}),
            websocket
        )
    finally:
        manager.disconnect(websocket)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)