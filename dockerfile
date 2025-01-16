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
    build-essential \
    espeak-ng \
    libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install Hugging Face Transformers from GitHub repository
RUN pip install git+https://github.com/huggingface/transformers.git

# Clone the Bark repository and install it
RUN git clone https://github.com/suno-ai/bark && \
    cd bark && \
    pip install .

# Clone the TTS dependencies (if applicable)
# If you have a GitHub repo for TTS, uncomment and modify this:
# RUN git clone https://github.com/your-username/your-tts-repo.git /app/tts

# Copy the application code into the container
COPY app/ ./app

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to start the FastAPI application
CMD ["uvicorn", "app.main_file:app", "--host", "0.0.0.0", "--port", "8000"]
