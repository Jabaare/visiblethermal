# Face Recognition and Identification System

This project is a Python-based face recognition and identification system that compares a probe image against a gallery of known images to identify subjects. The system uses the `face_recognition` library to encode and match faces, supporting multiple face detection and confidence-based identification.

## Features
- **Multiple Face Detection**: Identify multiple faces in a single probe image.
- **Confidence Threshold**: Set a confidence level to ensure accurate identification.
- **Gallery Comparison**: Match probe images against a gallery of known faces.
- **Detailed Output**: Reports identified faces with confidence scores or indicates if no match was found.

## Requirements
- Python 3.x
- `face_recognition` library
- `opencv-python` (for image handling)
- `os` (for directory management)

### Install Dependencies
```bash
pip install face_recognition opencv-python

## Project Structure

project/
│
├── gallery/            # Folder for known face images
│   ├── alice.jpg
│   └── bob.jpg
│
├── probe.jpg           # Probe image to analyze
├── identify.py         # Main face recognition script
├── README.md           # Project documentation
└── LICENSE             # License file
