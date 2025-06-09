# Use an official Python runtime as a parent image
FROM python:3.13-rc-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir: Disables the cache, which is not needed in a container and keeps the image size smaller.
# --trusted-host pypi.python.org: Can help prevent SSL issues in some network environments.
RUN pip install --no-cache-dir --upgrade -r requirements.txt --trusted-host pypi.python.org

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Run uvicorn server
# --host 0.0.0.0: Makes the server accessible from outside the container.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 