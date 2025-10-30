# Start with an official Python 3.11 image
FROM python:3.11-slim

# Install system dependencies: git and git-lfs
RUN apt-get update && \
    apt-get install -y git git-lfs && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements file first (for better caching)
COPY requirements.txt requirements.txt

# Install python libraries
RUN pip install -r requirements.txt

# Copy all the project files into the container
COPY . .

# Download the large model file
RUN git lfs pull

# Expose the port Render expects
EXPOSE 10000

# The Start Command will be set in the Render dashboard