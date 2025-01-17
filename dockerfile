# Stage 1: Builder stage
FROM windowsservercore-ltsc2022 AS builder

# Set the working directory for the application code
WORKDIR /app

# Install system dependencies required for building the application
RUN powershell -Command \
    "Set-ExecutionPolicy Bypass -Scope Process -Force; \
     Invoke-WebRequest -Uri https://aka.ms/vs/17/release/vs_BuildTools.exe -OutFile vs_buildtools.exe; \
     Start-Process -Wait -FilePath vs_buildtools.exe -ArgumentList '--quiet --wait --norestart --nocache --add Microsoft.VisualStudio.Workload.VCTools'; \
     Remove-Item vs_buildtools.exe" && \
    pip install --upgrade pip setuptools

# Install additional dependencies for audio processing
RUN pip install \
    pipwin \
    sounddevice \
    pyaudio \
    ffmpeg-python \
    phonemizer \
    torch \
    transformers \
    scipy \
    munch

# Clone and install Bark into the container
RUN git clone https://github.com/suno-ai/bark /app/bark && \
    pip install /app/bark

# Copy application dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runner stage
FROM python:3.12.8-windowsservercore-1809 AS runner

# Set the working directory for the application code
WORKDIR /app

# Copy installed Python dependencies and application code
COPY --from=builder /app/ /app/

# Expose the port for FastAPI
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "app.main_file:app", "--host", "0.0.0.0", "--port", "8000"]
