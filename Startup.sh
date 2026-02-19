#!/bin/bash
# Startup script for Azure App Service
# Save this as: Startup.sh

echo "Starting application..."
pip install -r requirements.txt
gunicorn --bind 0.0.0.0 --workers 1 --timeout 0 --access-logfile - --error-logfile - app:app
