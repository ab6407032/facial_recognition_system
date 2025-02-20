
# Facial Recognition System with Landmark Detection

This project is a real-time facial recognition system that uses OpenCV for face detection and landmark detection. The system captures video from your webcam, detects faces in real-time, and marks facial landmarks using a pre-trained LBF model.

## Features

- Real-time face detection using Haar Cascade.
- Real-time facial landmark detection using OpenCV's FacemarkLBF.
- Live video feed display with detected faces and landmarks highlighted.
- Integrated GUI using Tkinter.

## Prerequisites

Make sure you have the following installed:

- Python 3.6+
- OpenCV (with contrib modules)
- NumPy
- Pillow

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/facial_recognition_system.git
   cd facial_recognition_system
   ```

2. **Create a virtual environment (optional but recommended)**:
   ```bash
   python -m venv face_rec_env
   source face_rec_env/bin/activate  # On Windows use `face_rec_env\Scripts\activate`
   ```

3. **Install the required packages**:
   ```bash
   pip install opencv-contrib-python numpy Pillow
   ```

4. **Download the LBF model**:
   Download the `lbfmodel.yaml` file from [here](https://github.com/kurnianggoro/GSOC2017/tree/master/data) and place it in the root directory of the project.

## Usage

1. **Run the facial recognition application**:
   ```bash
   python face_recognition_ui.py
   ```

2. The application will open a window showing the live video feed from your webcam. Detected faces will be outlined with rectangles, and facial landmarks will be marked with red circles.

## Project Structure

```
facial_recognition_system/
│
├── face_recognition_ui.py   # Main script for the facial recognition system
├── lbfmodel.yaml            # Pre-trained LBF model for landmark detection
├── README.md                # Project documentation
└── requirements.txt         # List of required Python packages
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [OpenCV](https://opencv.org/) - Open Source Computer Vision Library
- [FacemarkLBF Model](https://github.com/kurnianggoro/GSOC2017/tree/master/data) - Pre-trained model for facial landmark detection
