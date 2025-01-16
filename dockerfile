# Use a lightweight Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install necessary system dependencies and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    python3-dev \
    libasound2-dev \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0 \
    fmpeg \
    build-essential \
    alsa-utils \
    ffmpeg \
    espeak-ng \
    libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt . 

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional necessary Python libraries for TTS and other tasks
RUN pip install --no-cache-dir phonemizer torch transformers scipy munch

# Install transformers and Bark
RUN pip install git+https://github.com/huggingface/transformers.git
RUN git clone https://github.com/suno-ai/bark && cd bark && pip install .


# Copy the application code into the container
COPY app/ ./app

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to start the FastAPI application
CMD ["uvicorn", "app.main_file:app", "--host", "0.0.0.0", "--port", "8000"]
