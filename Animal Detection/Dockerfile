FROM python:3.8-slim-buster

WORKDIR /app

COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Set the environment variable to tell Flask that we are running in production mode
ENV FLASK_ENV=production

# Expose port 8000 for the Flask app
EXPOSE 8000

# Start the Flask app
CMD ["python", "app.py"]
