FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Install supervisor to run two processes
RUN pip install supervisor

# Create supervisor config
RUN mkdir -p /etc/supervisor/conf.d
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

EXPOSE 8501 8000

CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]