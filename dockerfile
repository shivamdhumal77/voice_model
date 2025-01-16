# Stage 1: Builder stage
FROM python:3.9-slim AS builder

# Set the working directory for the application code
WORKDIR /app

# Install system dependencies required for building the application
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    python3-dev \
    libasound2-dev \
    ffmpeg \
    espeak-ng \
    libsndfile1 \
    build-essential \
    alsa-utils \
    && rm -rf /var/lib/apt/lists/*

# Create a Python virtual environment
RUN python3 -m venv /app/venv

# Set environment variables to use the virtual environment
ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies into the virtual environment
RUN pip install --no-cache-dir -r requirements.txt

# Install additional necessary Python libraries for TTS and other tasks into the virtual environment
RUN pip install --no-cache-dir phonemizer torch transformers scipy munch sounddevice

RUN apt-get update
RUN apt-get install libasound-dev libportaudio2 libportaudiocpp0 portaudio19-dev -y
RUN pip install pyaudio sounddevice
RUN apt-get install gcc -y

# Install transformers and Bark into the virtual environment
RUN pip install git+https://github.com/huggingface/transformers.git
RUN git clone https://github.com/suno-ai/bark /app/bark && \
    cd /app/bark && \
    pip install .

# Stage 2: Runner stage
FROM python:3.9-slim AS runner

# Set the working directory for the application code
WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /app/venv /app/venv

# Copy the application code into the container
COPY app/ ./app

# Set environment variables to use the virtual environment
ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Expose the port for FastAPI
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "app.main_file:app", "--host", "0.0.0.0", "--port", "8000"]
