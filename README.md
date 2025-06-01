# Room Acoustics Simulator

An interactive Python application that simulates sound wave propagation in 2D rooms with real-time visualization and acoustic analysis capabilities. This simulator is designed for computational physics applications in room acoustics.

## Features

- Interactive room design with corner placement and customizable layouts
- Multiple sound sources with adjustable frequency (125-8000 Hz) and amplitude
- Microphone placement for acoustic measurements at multiple points
- Real-time wave propagation visualization with customizable view settings
- Material selection for walls, ceiling, and floor with realistic acoustic properties
- Room preset configurations for different acoustic environments
- Height adjustment for 3D room modeling
- Acoustic analysis including:
  - Room Impulse Response (RIR)
  - RT60 calculation (reverberation time)
  - Frequency Response Analysis
  - Early Decay Time
  - Speech and Music Clarity (C50, C80)
  - Mean Free Path
  - Critical Distance
- 2D and 3D room visualization with detailed acoustic plots
- Ray tracing for accurate sound reflection simulation

## Controls

- **Room Controls**
  - H: Height Mode (adjust room height)
  - M: Room Mode (create/edit room layout)
  - X: Mic Mode (place measurement microphones)
  - P: Cycle through room presets
  - ESC: Clear room drawing
  - RETURN: Complete room drawing
  
- **Source Controls**
  - F: Activate single sound source
  - SPACE: Activate all sound sources
  - S: Add new sound source
  - Tab: Cycle through sound sources
  - Up/Down Arrow: Adjust frequency (Hz)
  - Left/Right Arrow: Adjust amplitude
  
- **Analysis Controls**
  - C: Calculate room acoustics
  - 2: 2D room view
  - 3: 3D room view

## Requirements

- Python 3.x
- pygame
- numpy
- pyroomacoustics
- matplotlib

## Room Presets

The application comes with several built-in room presets:
- Small rectangular room
- Hall/auditorium
- Home theater
- Recording studio
- C-shaped Room

Each preset includes optimal source and microphone positions for different acoustic scenarios.

## Materials

The simulator features realistic acoustic materials with frequency-dependent absorption properties:
- **Walls:** Concrete, brick, glass, wooden lining, ceramic tiles, marble, and more
- **Ceiling:** Acoustic tiles, concrete, wooden materials
- **Floor:** Carpet, concrete, linoleum, ceramic, marble

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

## Usage Example

1. Press 'P' to cycle through room presets or 'M' to create your own room design
2. Add sound sources with 'S' and place microphones with 'X'
3. Select materials for walls, ceiling, and floor from the dropdown menus
4. Press 'C' to calculate acoustic properties
5. View room visualization in 2D ('V') or 3D ('3') mode
