# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
# We copy it separately to leverage Docker layer caching
COPY ../requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend application code and the trained model into the container
COPY . .

# Command to run the app using uvicorn
# It will be accessible on port 10000 within the container
# Render will automatically handle mapping this port to the outside world
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]