FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080
EXPOSE 8501

COPY docker/start.sh /app/docker/start.sh
RUN chmod +x /app/docker/start.sh

CMD ["/app/docker/start.sh"]
