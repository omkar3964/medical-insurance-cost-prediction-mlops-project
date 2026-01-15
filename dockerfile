FROM python:3.11-slim

# Ensure logs appear immediately
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy and install only Flask dependencies first (layer caching)
COPY flask_app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy Flask application package
COPY flask_app /app/flask_app

EXPOSE 5000

#local
# CMD ["python", "app.py"]

# Production WSGI server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "flask_app.app:app"]
