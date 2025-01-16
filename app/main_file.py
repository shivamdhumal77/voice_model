from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import subprocess
import io
import numpy as np
import torch
import warnings
from transformers import AutoProcessor, BarkModel
from tokenizers import Tokenizer
import whisper
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

warnings.filterwarnings(
    "ignore",
    message="torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.",
)

# Initialize FastAPI app
app = FastAPI()

# Initialize models and services
stt = whisper.load_model("base.en")

class TextToSpeechService:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained("suno/bark-small")
        self.model = BarkModel.from_pretrained("suno/bark-small")
        self.model.to(self.device)
        self.tokenizer = Tokenizer.from_pretrained("bert-base-uncased")

    def synthesize(self, text: str, voice_preset: str = "v2/en_speaker_1"):
        inputs = self.processor(text, voice_preset=voice_preset, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            audio_array = self.model.generate(**inputs, pad_token_id=10000)
        audio_array = audio_array.cpu().numpy().squeeze()
        sample_rate = self.model.generation_config.sample_rate
        return sample_rate, audio_array

    def long_form_synthesize(self, text: str, voice_preset: str = "v2/en_speaker_1"):
        pieces = []
        silence = np.zeros(int(0.25 * self.model.generation_config.sample_rate))
        tokens = self.tokenizer.encode(text)
        sentences = tokens.tokens
        for sent in sentences:
            sample_rate, audio_array = self.synthesize(sent, voice_preset)
            pieces += [audio_array, silence.copy()]
        return self.model.generation_config.sample_rate, np.concatenate(pieces)

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

# Utility Functions
def record_audio_with_ffmpeg(output_file="recording.wav", silence_duration=3):
    """
    Continuously records audio until silence is detected for a specified duration.
    """
    try:
        command = [
            "ffmpeg", "-y", "-f", "alsa", "-i", "default",
            "-af", f"silencedetect=n=-30dB:d={silence_duration}",
            output_file
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

@app.post("/process")
async def process_audio():
    """
    Processes audio from the microphone, transcribes it, generates a response, synthesizes speech,
    and provides a StreamingResponse with the audio.
    """
    # Record audio using FFmpeg
    audio_file = record_audio_with_ffmpeg()

    # Transcribe the audio file
    text = transcribe_audio(audio_file)
    print(f"You: {text}")

    # Get LLM response
    response = get_llm_response(text)
    print(f"Assistant: {response}")

    # Synthesize speech
    sample_rate, audio_array = tts.long_form_synthesize(response)

    # Prepare audio for streaming
    audio_stream = io.BytesIO()
    audio_stream.write(audio_array.astype(np.float32).tobytes())
    audio_stream.seek(0)

    return StreamingResponse(
        audio_stream,
        media_type="audio/wav",
        headers={"Content-Disposition": "inline; filename=response.wav"}
    )
