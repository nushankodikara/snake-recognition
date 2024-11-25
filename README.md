# 🐍 Snake Species Classifier

A deep learning-based web application for identifying snake species from images. The project consists of a FastAPI backend service for predictions and a Streamlit frontend for user interaction.

## Features

- Upload and classify snake images
- Get detailed species information including:
  - Scientific name and taxonomy
  - Geographic distribution
  - Venomous status
  - Confidence scores
- View alternative predictions
- Dataset statistics and model information

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd snake-species-classifier
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python3.9 -m venv venv
source venv/bin/activate
```


3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the API Server

Start the FastAPI server with:
```bash
uvicorn api.main:app --reload
```

The API will be available at `http://localhost:8000`

API Endpoints:
- `GET /`: Root endpoint with API information
- `GET /species`: List all snake species in the database
- `POST /predict`: Upload and classify a snake image

### Running the Streamlit Application

Start the Streamlit web interface with:
```bash
streamlit run streamlit/app.py
```
The application will open in your default web browser.

## Project Structure

- `api/`: FastAPI backend service
- `streamlit/`: Streamlit frontend application
- `data/`: Data processing scripts and species information
- `models/`: Trained model files (using Git LFS)
- `requirements.txt`: Project dependencies

## Technical Details

### Model Architecture
- Based on EfficientNetB4
- Trained on a diverse dataset of snake images
- Supports multiple snake species classification
- Input image size: 480x480 pixels

### Data Processing
- Images are automatically resized and normalized
- Predictions are mapped to valid class IDs
- Confidence scores are provided for top predictions

## Important Notes

⚠️ **Safety Warning**
- This tool is for educational purposes only
- Do not rely solely on this application for snake identification
- Always maintain a safe distance from snakes
- Contact professional snake handlers when needed

## Dependencies

The project requires several Python packages, which are listed in `requirements.txt`. Key dependencies include:
- TensorFlow and Keras for model inference
- FastAPI for the backend API
- Streamlit for the web interface
- OpenCV and Pillow for image processing
- NumPy and Pandas for data handling

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for any improvements or bug fixes.