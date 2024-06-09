# Facial-Expression-Driven-Document-Navigation 

Computer vision to enable users to navigate through PDF documents using only their eye movements and blinks. By closing their eyes for three seconds, users can advance to the next page, previous of the document [general-navigation]

## Features

- **Eye Detection and Tracking**: Leverages facial landmarks to detect and track the user's eye movements.
- **Blink Navigation**: Moves to the next page of the PDF when a blink lasting three seconds is detected.
- **Real-Time Feedback**: Provides real-time visual feedback on eye status (open or closed) and the number of blinks detected.

## Prerequisites

Before you run the application, ensure you have the following installed:
- Python 3.6 or higher
- OpenCV
- Dlib
- Imutils
- PyAutoGUI
- NumPy
- SciPy

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/[your-username]/eye-navigate-pdf-reader.git
2. Change directory to the project folder:
   ```bash
   cd Facial-Expression-Driven-Document-Navigation 
3. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt

## Usage Guidelines

* Execute the following command in your terminal to launch the application:
   ```bash
   python eye_navigate_pdf_reader.py
Prior to initiating the program, ensure that your webcam is operational and correctly configured to capture video input.

## Architectural Overview

The application employs a robust methodology to facilitate document navigation through visual cues:
1. **Facial Recognition**: Leverages real-time video analysis to locate and track the user's facial features.
2. **Precision Eye Tracking**: Employs facial landmarks to determine eye positions and calculate the Eye Aspect Ratio (EAR) for blink detection.
3. **Automated Blink Detection**: Analyzes changes in EAR to detect blinks and interpret prolonged closures as page-turning commands.
4. **Command Execution**: Simulates keyboard inputs (right arrow key) to navigate PDFs, effectively translating blinks into actionable commands.

## License

This project is licensed.

## Contact Information

- Name: Jaiteg Chahal
- Email: jchahal@berkeley.edu
- Project Repository: https://github.com/jaitegchahal123/Facial-Expression-Driven-Document-Navigation
