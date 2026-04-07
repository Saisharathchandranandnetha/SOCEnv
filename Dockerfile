# Dockerfile – reproducible container for AI SOC Gym
# ---------------------------------------------------
FROM python:3.11-slim

# Create a non‑root user (best practice for hackathons)
RUN useradd -m appuser
WORKDIR /home/appuser

# System dependencies (very small set)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /home/appuser

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Switch to the non‑root user
USER appuser

# Default command – run a quick metrics evaluation (can be overridden)
CMD ["python", "metrics.py"]
