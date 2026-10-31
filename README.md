<div align="center">

# Emotion-Based Music Playlist Generator

### Music recommendations from facial expressions

Python · TensorFlow/Keras · OpenCV · Flask · Spotipy

</div>

---

## Overview

This project uses a webcam to classify facial expressions and recommend songs that match the predicted emotion. For example, a sad expression leads to sad-song recommendations.

The application combines a facial-expression model trained on FER2013 with a Flask interface and emotion-specific song lists. Spotify playlist utilities are used to refresh the recommendations.

## How It Works

1. The webcam captures a live video feed.
2. A face detector locates the face in each frame.
3. The face image is converted to grayscale and resized for the model.
4. A convolutional neural network predicts an expression class.
5. The application selects the corresponding song list and displays recommendations.

```text
Webcam → Face detection → Image preprocessing
                              ↓
                     Expression prediction
                              ↓
                  Song list → Flask interface
```

## Features

- Webcam-based facial-expression classification
- Seven expression classes: angry, disgusted, fearful, happy, neutral, sad, and surprised
- Song recommendations matched to the predicted expression
- Spotify playlist utilities for updating song lists
- Threaded webcam processing

## Technology

| Component | Tools |
| :--- | :--- |
| Model development | TensorFlow, Keras |
| Face detection and image processing | OpenCV |
| Web application | Flask, HTML, CSS |
| Playlist retrieval | Spotipy |
| Song data | CSV files |

## Dataset and Preprocessing

The model uses the FER2013 facial-expression dataset.

Images are prepared as 48 × 48 grayscale inputs and normalized before training. The documented training setup uses Keras `ImageDataGenerator` with a batch size of 64.

Class imbalance is a consideration when evaluating the model. Overall accuracy alone does not show how reliably it distinguishes each expression.

## Model Architecture

The documented architecture is a sequential convolutional neural network with:

- Convolutional layers using ReLU activation and 32–128 filters
- Max-pooling layers with a 2 × 2 pool size
- Dropout layers to reduce overfitting
- A final softmax layer for seven-class prediction

The model uses the Adam optimizer and categorical cross-entropy loss.

## Training and Evaluation

The original project notes report approximately 66% accuracy after 75 epochs, with training taking around 13 hours on the local machine used.

The notes do not identify whether that figure is training, validation, or test accuracy, so it should be treated as a preliminary result.

A more complete evaluation would include:

- Results on a clearly identified held-out test set
- Precision, recall, and F1-score for each class
- A confusion matrix
- Examples of incorrect predictions

## Repository Guide

| File | Purpose |
| :--- | :--- |
| `app.py` | Flask application entry point and routes |
| `camera.py` | Webcam processing, expression prediction, and recommendation logic |
| `train.py` | Model training |
| `model.h5` | Saved model |
| `Spotipy.py` | Spotify playlist retrieval utilities |
| `utils.py` | Threaded webcam utilities |
| Emotion-specific CSV files | Song lists used for recommendations |
| HTML and CSS files | Application interface |

## Running Locally

The application is intended to run locally with a webcam.

Setup documentation is being updated. The following items still need to be resolved or verified:

- Add a dependency file.
- Place the application template at `templates/index.html`, as expected by `app.py`.
- Verify the face-detector assets and their paths.

Once dependencies, assets, and template paths are configured, start the application with:

```bash
python app.py
```

Spotify credentials are needed when refreshing playlists through `Spotipy.py`. Keep credentials outside committed source files, using environment variables or another local configuration method.

## Limitations

- Facial expressions do not reliably establish a person's actual emotional state.
- Lighting, camera position, and image quality can affect predictions.
- Recommendations match songs to an expression category; they do not learn individual listening preferences.
- The original accuracy figure needs a documented evaluation before it can support performance claims.

## Next Steps

- Complete and verify installation instructions
- Add a dependency file and check required assets
- Add screenshots and a short demonstration
- Document per-class model performance
- Allow users to select or correct the predicted emotion
