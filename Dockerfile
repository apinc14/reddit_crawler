# Use the official Python base image
FROM python:3.10
# Set the working directory inside the container
WORKDIR /app
# Copy the Python script into the container
COPY  requirements.txt crawler.py ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
# Install application dependencies
EXPOSE 3306
# Run the Python script when the container starts
CMD [ "python", "createTable.py" ]
CMD ["python", "crawler.py"]
