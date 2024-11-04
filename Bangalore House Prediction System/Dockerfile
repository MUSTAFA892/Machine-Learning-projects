FROM python:3.11.4

# Set environment variable for the port
ENV PORT=5000

# Copy the current directory contents into the /app directory in the container
COPY . /app

# Set the working directory to /app
WORKDIR /app

# Install the required Python packages
RUN pip install -r requirements.txt

# Expose the port
EXPOSE $PORT

# Command to run your application (e.g., Flask or FastAPI)
CMD ["python", "main.py"]
