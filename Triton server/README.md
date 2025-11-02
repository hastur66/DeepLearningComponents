# Triton inference server demo

NVIDIA Triton Inference Server for serving deep learning models with high performance and scalability.

## Quick Start

1. **Start server**: `docker-compose up -d`
2. **Run inference**: `python example/triton_inference.py`

## Structure

- `docker-compose.yml` - Triton server configuration
- `models/` - Model repository with DenseNet ONNX model
- `example/` - Python inference clients

## Endpoints

- HTTP: `localhost:8000`
- gRPC: `localhost:8001`
- Metrics: `localhost:8002`