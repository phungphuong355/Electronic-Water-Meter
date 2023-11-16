FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY setup.txt .

# Install dependencies
RUN pip install --no-cache-dir -r setup.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


# Copy the rest of the application files
COPY . .