# FROM python:3.11-slim-bookworm
# WORKDIR /app
# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt
# COPY main.py ./
# CMD ["python","-u","main.py"]


FROM python:3.11-slim

# System deps for lxml/readability; remove if you don't need them
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl libxml2-dev libxslt1-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

ENV PYTHONUNBUFFERED=1
CMD ["python", "-u", "main.py"]
