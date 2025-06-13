#!/bin/bash

echo "Starting FastAPI app..."
uvicorn main:app --host 0.0.0.0 --port 8000
