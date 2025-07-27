FROM python:3.10-slim

# Instalar dependencias del sistema incluyendo las necesarias para OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libmariadb-dev-compat \
    libmariadb-dev \
    # Dependencias adicionales para OpenCV en entornos headless
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Configurar variables de entorno para OpenCV headless
ENV DEBIAN_FRONTEND=noninteractive
ENV QT_X11_NO_MITSHM=1
ENV DISPLAY=:99

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]