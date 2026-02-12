# Docker Setup for NTPN GUI

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Build and run
docker-compose up --build

# Run in detached mode
docker-compose up -d

# Stop the container
docker-compose down
```

The app will be available at http://localhost:8501

### Using Docker directly

```bash
# Build the image
docker build -t ntpn-gui .

# Run the container
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  ntpn-gui

# Run in detached mode
docker run -d -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  --name ntpn-gui \
  ntpn-gui

# Stop the container
docker stop ntpn-gui
docker rm ntpn-gui
```

## Volume Mounts

The Docker setup mounts two directories:
- `./data` - Your neural data files (persists across container restarts)
- `./models` - Trained model files (persists across container restarts)

This ensures your data and trained models are not lost when the container is stopped.

## Development

For development with live code reloading:

```bash
docker run -p 8501:8501 \
  -v $(pwd):/app \
  ntpn-gui
```

This mounts the entire project directory, so code changes will be reflected immediately.

## Troubleshooting

### Port already in use
If port 8501 is already in use, change the port mapping:
```bash
docker run -p 8080:8501 ntpn-gui
```
Then access at http://localhost:8080

### Container won't start
Check logs:
```bash
docker logs ntpn-gui
```

### Out of memory
TensorFlow can be memory-intensive. Increase Docker's memory limit:
- Docker Desktop: Settings → Resources → Memory

## Building for Production

For a production build without dev dependencies:

```bash
docker build -t ntpn-gui:prod --target production .
```
