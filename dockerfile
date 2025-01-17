# Stage 1: Builder stage
FROM nvidia/cuda:12.2.0-cudnn8-devel-ubuntu20.04 AS builder

# Set the working directory for the application code
WORKDIR /app

# Install system dependencies required for building the application
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3.9 \
    python3.9-venv \
    python3.9-dev \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    libasound2-dev \
    ffmpeg \
    espeak-ng \
    libsndfile1 \
    build-essential \
    gcc \
    && ln -sf /usr/bin/python3.9 /usr/bin/python3 && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN python3 -m venv /app/venv && \
    /app/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel

# Set environment variables to use the virtual environment
ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy the requirements file into the container
COPY requirements.txt ./

# Install Python dependencies into the virtual environment
RUN pip install --no-cache-dir -r requirements.txt

# Install additional Python libraries for TTS and audio tasks
RUN pip install --no-cache-dir \
    phonemizer \
    torch \
    transformers \
    scipy \
    munch \
    sounddevice \
    pyaudio

# Clone and install Bark into the virtual environment
RUN git clone https://github.com/suno-ai/bark /app/bark && \
    pip install /app/bark

# Stage 2: Runner stage
FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu20.04 AS runner

# Set the working directory for the application code
WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /app/venv /app/venv

# Copy the application code into the container
COPY app/ ./app

# Set environment variables to use the virtual environment
ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install runtime dependencies for audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    libasound2-dev \
    ffmpeg \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Expose the port for FastAPI
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "app.main_file:app", "--host", "0.0.0.0", "--port", "8000"]
