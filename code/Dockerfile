FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the webapp directory
COPY webapp/ ./webapp/

# Set environment variables
ENV FLASK_APP=webapp/app.py
ENV FLASK_ENV=production

# Expose port
EXPOSE 5000

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--chdir", "webapp", "app:app"] 