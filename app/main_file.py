from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
import threading
import subprocess
import numpy as np
import whisper
from queue import Queue
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from text_to_speech import TextToSpeechService
import io
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Initialize models and services
stt = whisper.load_model("base.en")
tts = TextToSpeechService()

# Initialize LLM (Athene-v2)
llm = ChatOpenAI(
    api_key="ollama",
    base_url="https://sunny-gerri-finsocialdigitalsystem-d9b385fa.koyeb.app/v1",
    model="athene-v2"
)

# Define conversation prompt
template = """
You are a helpful and friendly AI assistant. You are polite, respectful, and aim to provide concise responses of less 
than 20 words.

The conversation transcript is as follows:
{history}

And here is the user's follow-up: {input}

Your response:
"""
PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
chain = ConversationChain(
    prompt=PROMPT,
    verbose=False,
    memory=ConversationBufferMemory(ai_prefix="Assistant:"),
    llm=llm,
)

# Utility functions
def record_audio_with_ffmpeg(output_file="recording.wav", duration=10):
    """
    Records audio using FFmpeg and saves it to a file.
    """
    try:
        command = [
            "ffmpeg", "-y", "-f", "alsa", "-i", "default",
            "-t", str(duration), output_file
        ]
        subprocess.run(command, check=True)
        return output_file
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Audio recording failed: {e}")

def transcribe_audio(file_path: str) -> str:
    """
    Transcribes the given audio file using the Whisper model.
    """
    try:
        result = stt.transcribe(file_path, fp16=False)
        return result["text"].strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

def get_llm_response(text: str) -> str:
    """
    Generates a response using the Athene-v2 LLM.
    """
    try:
        response = chain.predict(input=text)
        if response.startswith("Assistant:"):
            return response[len("Assistant:"):].strip()
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM response generation failed: {e}")

def synthesize_speech(text: str) -> io.BytesIO:
    """
    Synthesizes speech and returns the audio as a file in memory (WAV format).
    """
    try:
        sample_rate, audio_array = tts.long_form_synthesize(text)
        audio_io = io.BytesIO()
        audio_io.write(audio_array.tobytes())
        audio_io.seek(0)  # Reset file pointer
        return audio_io
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text-to-speech synthesis failed: {e}")

# Data model
class Message(BaseModel):
    text: str

@app.on_event("startup")
async def startup_event():
    """
    On startup, send the initial "Hello" text.
    """
    initial_text = "Hello"
    print(f"You: {initial_text}")
    response = get_llm_response(initial_text)
    print(f"Assistant: {response}")
    audio_io = synthesize_speech(response)
    # Just an example, you may want to save or play the audio here

@app.post("/process")
async def process_audio(background_tasks: BackgroundTasks):
    """
    Processes audio from the microphone, transcribes it, generates a response, and synthesizes speech.
    """
    print("Listening... Speak into the microphone.")
    
    # Record audio using FFmpeg
    audio_file = record_audio_with_ffmpeg()

    # Process the recorded audio
    text = transcribe_audio(audio_file)
    print(f"You: {text}")

    response = get_llm_response(text)
    print(f"Assistant: {response}")

    audio_io = synthesize_speech(response)
    return StreamingResponse(audio_io, media_type="audio/wav")
