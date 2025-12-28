# Virtual_Calculator
# Gesture-Controlled Hand Calculator using Computer Vision

# Project Overview
This project is a real-time gesture-controlled virtual calculator that allows users to perform mathematical operations using hand gestures captured via a webcam. The system uses MediaPipe for hand landmark detection and OpenCV for real-time video processing.

Users can input numbers using finger-count gestures and select arithmetic operators through intuitive hand movements, enabling touchless human–computer interaction.


# Features
- Real-time hand gesture recognition
- Touchless number input using finger counting (0–9)
- Operator selection using gesture-based interaction
- Supports addition, subtraction, multiplication, and division
- Safe evaluation of mathematical expressions
- Visual feedback for gestures and calculator state


# Technologies Used
Programming Language: Python
Libraries:
  - OpenCV
  - MediaPipe
  - NumPy
Concepts:
  - Computer Vision
  - Hand Landmark Detection
  - Gesture Recognition
  - Human–Computer Interaction (HCI)


# System Workflow
1. Webcam captures live video frames
2. MediaPipe detects hand landmarks
3. Finger-count logic interprets numeric gestures
4. Gesture-based selection chooses operators
5. Expression is safely evaluated
6. Result is displayed on screen in real time


# Gesture Controls
Gesture and it's Functions
1. Finger count (0–9) --> Number input
2. Index finger at top panel --> Operator selection
3. Thumb + index close --> Move / ignore input
4. Operator `=` --> Evaluate expression
5. Operator `C` --> Clear input
