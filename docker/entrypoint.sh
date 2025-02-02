#!/bin/bash
set -e

# Start the API service
python3 -m src.api

# Keep the container running
exec "$@"
