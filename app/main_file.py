from fastapi import FastAPI, HTTPException, BackgroundTasks, Response
from fastapi.responses import StreamingResponse
import threading
import time
import numpy as np
import whisper
import sounddevice as sd
from queue import Queue
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from test_to_speech_1 import TextToSpeechService
import io

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
def record_audio(stop_event, data_queue, silence_duration=3, silence_threshold=0.01):
    """
    Records audio from the microphone and stops if silence is detected for a specified duration.
    """
    silence_counter = 0

    def callback(indata, frames, time, status):
        nonlocal silence_counter
        if status:
            print("Audio stream status:", status)
        
        amplitude = np.abs(np.frombuffer(indata, dtype=np.int16)).mean()
        if amplitude < silence_threshold * 32768:  # Silence condition
            silence_counter += 1
        else:
            silence_counter = 0  # Reset counter if sound is detected
        
        data_queue.put(bytes(indata))
        if silence_counter > (silence_duration * 10):  # Stop if silence persists
            stop_event.set()

    with sd.RawInputStream(
        samplerate=16000, dtype="int16", channels=1, callback=callback
    ):
        while not stop_event.is_set():
            time.sleep(0.1)

def transcribe_audio(audio_np: np.ndarray) -> str:
    """
    Transcribes the given audio data using the Whisper model.
    """
    try:
        result = stt.transcribe(audio_np, fp16=False)
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
        sd.write(audio_io, audio_array, sample_rate, format="WAV")
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
    play_audio(audio_io)

@app.post("/process")
async def process_audio(background_tasks: BackgroundTasks):
    """
    Processes audio from the microphone, transcribes it, generates a response, and synthesizes speech.
    """
    print("Listening... Speak into the microphone.")
    data_queue = Queue()
    stop_event = threading.Event()

    # Record audio in a background thread
    recording_thread = threading.Thread(
        target=record_audio, args=(stop_event, data_queue)
    )
    recording_thread.start()
    recording_thread.join()

    # Process the recorded audio
    audio_data = b"".join(list(data_queue.queue))
    if not audio_data:
        raise HTTPException(status_code=400, detail="No audio detected. Please try again.")
    
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    text = transcribe_audio(audio_np)
    print(f"You: {text}")

    response = get_llm_response(text)
    print(f"Assistant: {response}")

    audio_io = synthesize_speech(response)
    background_tasks.add_task(play_audio, audio_io)
    return StreamingResponse(audio_io, media_type="audio/wav")

