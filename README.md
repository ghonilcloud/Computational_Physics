# Room Acoustics Simulator

An interactive Python application that simulates sound wave propagation in 2D rooms with real-time visualization and acoustic analysis capabilities.

## Features

- Interactive room design with corner placement
- Multiple sound sources with adjustable frequency and amplitude
- Microphone placement for acoustic measurements
- Real-time wave propagation visualization
- Acoustic analysis including:
  - Room Impulse Response (RIR)
  - RT60 calculation
  - Frequency Response Analysis
  - Early Decay Time
  - Speech and Music Clarity (C50, C80)
  - Mean Free Path
  - Critical Distance
- 2D and 3D room visualization

## Controls

- **Room Controls**
  - H: Height Mode
  - M: Room Mode
  - X: Mic Mode
  
- **Source Controls**
  - F: Single Source
  - SPACE: All Sources
  - S: Add Source
  
- **Analysis Controls**
  - C: Calculate
  - V: 2D View
  - 3: 3D View

## Requirements

- Python 3.x
- pygame
- numpy
- pyroomacoustics
- matplotlib

## Installation

1. Clone this repository
2. Install the required packages:
```
pip install pygame numpy pyroomacoustics matplotlib
```
3. Run the application:
```
python Project.py
```
