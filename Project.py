import numpy as np
import pygame
import sys
from dataclasses import dataclass
from typing import List
import colorsys
from math import floor
import pyroomacoustics as pra
from numpy import hamming
import matplotlib.pyplot as plt  # Add this line

@dataclass
class SoundSource:
    x: int
    y: int
    frequency: float = 440.0
    amplitude: float = 1.0
    active: bool = False
    color: tuple = (255, 255, 0)

@dataclass
class MeasurementPoint:
    x: int
    y: int
    readings: List[float]
    max_readings: int = 100

@dataclass
class Microphone:
    x: int
    y: int
    color: tuple = (255, 192, 203)  # Pink color for microphone

# Keybinds:
# F - Activate sound source
# Up/Down Arrow - Adjust frequency (Hz)
# Left/Right Arrow - Adjust amplitude
# S - Add sound source
# Tab - Cycle through sound sources

# RETURN - Complete room drawing
# ESC - Clear room drawing
# M - Enter room drawing mode
# Left Click - Place room corners (when in room drawing mode)

# Room dimensions (meters)
<<<<<<< HEAD
ROOM_WIDTH = 15.0
ROOM_HEIGHT = 15.0
=======
ROOM_WIDTH = 10.0
ROOM_HEIGHT = 10.0
>>>>>>> 03cd12879a6de7f74b97c478919f03b0f3719860

# Simulation parameters
dx = 0.1
dt = 0.5 * dx / 343.0
c = 343.0

# Grid size
nx = int(ROOM_WIDTH / dx)
ny = int(ROOM_HEIGHT / dx)

# Initialize wave field
wave = np.zeros((nx, ny))
wave_prev = np.zeros((nx, ny))

# Initial source position
source_x, source_y = nx // 6, ny // 6

# Define wall types
class WallType:
    NONE = 0
    REFLECTIVE = 1
    ABSORPTIVE = 2
    PARTIAL = 3

# Modify walls array to store wall types
walls = np.zeros((nx, ny), dtype=int)  # 0 = no wall, 1 = reflective, 2 = absorptive

# Pygame initialization
pygame.init()
screen_size = 800
<<<<<<< HEAD
bottom_panel_height = 100  # Height for bottom control panel
screen = pygame.display.set_mode((screen_size, screen_size + bottom_panel_height))
pygame.display.set_caption('Sound Wave Propagation')
=======
bar_height = 60  # Increased height for brush selection
screen = pygame.display.set_mode((screen_size, screen_size + bar_height))
pygame.display.set_caption('Sound Wave Propagation')  # Fixed method name
>>>>>>> 03cd12879a6de7f74b97c478919f03b0f3719860
clock = pygame.time.Clock()

# Colors
WALL_COLORS = {
    WallType.NONE: (128, 0, 0),  # Dark red background
    WallType.REFLECTIVE: (128, 0, 0),  # Match background red
    WallType.ABSORPTIVE: (128, 0, 0),  # Match background red
    WallType.PARTIAL: (128, 0, 0)      # Match background red
}

SOURCE_COLORS = [
    (255, 255, 0),  # Yellow
    (0, 255, 255),  # Cyan
    (255, 0, 255),  # Magenta
    (255, 128, 0)   # Orange
]

MEASUREMENT_COLOR = (255, 0, 255)
<<<<<<< HEAD
GRID_COLOR = (40, 40, 40)  # Darker gray for better contrast
INTENSITY_LINE_COLOR = (220, 220, 220)  # Lighter gray for text and lines
=======
GRID_COLOR = (200, 200, 200)
INTENSITY_LINE_COLOR = (100, 100, 100)
>>>>>>> 03cd12879a6de7f74b97c478919f03b0f3719860

# Scaling factor
scale_x = screen_size / nx
scale_y = screen_size / ny

# Sound activation
sound_active = False
decay_rate = 0.99

# Font for status bar
font = pygame.font.Font(None, 36)

# Initialize state
sources = []
microphones = []  # List to store microphone positions
selected_source_index = 0
mic_mode = False  # New variable to track microphone placement mode
measurement_points = []
show_grid = False
show_intensity_lines = False
use_rainbow_colormap = False

# Frequency and amplitude step sizes
FREQ_STEP = 10.0  # Hz per keypress
AMP_STEP = 0.1    # Amplitude change per keypress

<<<<<<< HEAD
# Add these constants after the other room/grid parameters
TEST_BOX_SIZE = 20  # Size of test boxes in grid units
TEST_BOX_SPACING = 10  # Space between boxes

FEET_TO_METERS = 0.3048  # Conversion factor
HEIGHT_STEP = 0.3  # Height change in meters per keypress
DEFAULT_ROOM_HEIGHT = 3.0 / FEET_TO_METERS  # Default height in feet (3 meters)

=======
>>>>>>> 03cd12879a6de7f74b97c478919f03b0f3719860
# Wall absorption coefficients
wall_coefficients = {
    WallType.NONE: 0.0,        # No effect on wave
    WallType.REFLECTIVE: 0.9,  # Reflects the wave (inverts amplitude)
    WallType.ABSORPTIVE: 0.1,   # Absorbs the wave (reduces amplitude)
    WallType.PARTIAL: 0.5       # Partially absorbs the wave
}

<<<<<<< HEAD
# Add acoustic materials
#These lines define acoustic materials for the walls, ceiling, and floor of the room using the 
# pyroomacoustics library. Each material is characterized by energy absorption and scattering.
=======
# Add these constants after the other room/grid parameters
TEST_BOX_SIZE = 20  # Size of test boxes in grid units
TEST_BOX_SPACING = 10  # Space between boxes

FEET_TO_METERS = 0.3048  # Conversion factor
DEFAULT_ROOM_HEIGHT = 10.0  # Default height in feet

# Add acoustic materials
>>>>>>> 03cd12879a6de7f74b97c478919f03b0f3719860
wall_material = pra.Material(energy_absorption=0.5, scattering=0.25)
ceiling_material = pra.Material(energy_absorption=0.5, scattering=0.25)
floor_material = pra.Material(energy_absorption=0.05, scattering=0.25)

# Add to the initialization section
room_height = DEFAULT_ROOM_HEIGHT  # Height in feet
height_input_active = False
height_input_text = str(DEFAULT_ROOM_HEIGHT)

# Add after the other global variables
current_rt60 = 0.8  # Default RT60 in seconds

def calculate_decay_rate(rt60):
    """Calculate appropriate decay rate based on RT60 value"""
    # We want amplitude to decay to 0.001 (-60dB) in rt60 seconds
    # So: decay_rate^(rt60/dt) = 0.001
    # Therefore: decay_rate = 0.001^(dt/rt60)
    return np.power(0.001, dt/rt60)

<<<<<<< HEAD
=======
def create_test_boxes():
    global walls
    # Clear existing walls
    walls.fill(WallType.NONE)
    
    # Calculate positions for two boxes
    start_x = nx // 4
    start_y = ny // 4
    
    # Create reflective box (left)
    for x in range(start_x, start_x + TEST_BOX_SIZE):
        for y in range(start_y, start_y + TEST_BOX_SIZE):
            if (x == start_x or x == start_x + TEST_BOX_SIZE - 1 or 
                y == start_y or y == start_y + TEST_BOX_SIZE - 1):
                walls[x, y] = WallType.REFLECTIVE
    
    # Create absorptive box (right)
    start_x += TEST_BOX_SIZE + TEST_BOX_SPACING
    for x in range(start_x, start_x + TEST_BOX_SIZE):
        for y in range(start_y, start_y + TEST_BOX_SIZE):
            if (x == start_x or x == start_x + TEST_BOX_SIZE - 1 or 
                y == start_y or y == start_y + TEST_BOX_SIZE - 1):
                walls[x, y] = WallType.ABSORPTIVE

>>>>>>> 03cd12879a6de7f74b97c478919f03b0f3719860
def update_wave():
    global wave, wave_prev
    wave_next = np.copy(wave)

    # Calculate current decay rate based on RT60
    current_decay = calculate_decay_rate(current_rt60)

<<<<<<< HEAD
    # Iterates through each sound source, and only generates noise is source is active.
    for source in sources:
        if source.active:  # Check individual source activation
            t = pygame.time.get_ticks() / 1000.0
            # Generates a wave at the source's position.
            wave[source.x, source.y] += source.amplitude * np.sin(2 * np.pi * source.frequency * t)
            # Source.amplitude is the wave amplitude/height/intensity
            # np.sin(2 * np.pi * source.frequency * t) is the wave frequency
            
    
    for x in range(1, nx - 1):
        for y in range(1, ny - 1):
            # Checks the wall type at the current position
            # and applies the appropriate reflection/absorption coefficient
=======
    for source in sources:
        if source.active:  # Check individual source activation
            t = pygame.time.get_ticks() / 1000.0
            wave[source.x, source.y] += source.amplitude * np.sin(2 * np.pi * source.frequency * t)
            
    for x in range(1, nx - 1):
        for y in range(1, ny - 1):
>>>>>>> 03cd12879a6de7f74b97c478919f03b0f3719860
            wall_type = walls[x, y]
            coefficient = wall_coefficients[wall_type]
            
            if wall_type == WallType.NONE:
<<<<<<< HEAD
                # Standard wave equation for propagation though 2D free space
                # It is a discretized 2D wave equation, 
                # Handles time evolution of the wave
                wave_next[x, y] = (2 * wave[x, y] - wave_prev[x, y] +
                                # c is speed of sound, dt is time step, dx is spatial step
                                (c * dt / dx) ** 2 *
                                # Gets the horizontal and vertical neighbors
=======
                # Standard wave equation for propagation
                wave_next[x, y] = (2 * wave[x, y] - wave_prev[x, y] +
                                (c * dt / dx) ** 2 *
>>>>>>> 03cd12879a6de7f74b97c478919f03b0f3719860
                                (wave[x + 1, y] + wave[x - 1, y] +
                                 wave[x, y + 1] + wave[x, y - 1] - 
                                 4 * wave[x, y]))
            elif wall_type == WallType.ABSORPTIVE:
                # Instant absorption for absorptive walls
                wave_next[x, y] = 0
            else:
                # Reflection for other wall types
                incident_wave = wave[x, y]
                wave_next[x, y] = coefficient * incident_wave

    # Apply RT60-based decay
    wave_next *= current_decay
    wave_prev[:], wave[:] = wave[:], wave_next[:]

# Replace the wall-related constants with room coordinate system
FEET_TO_METERS = 0.3048  # Conversion factor

class Room:
    def __init__(self):
        self.corners = []  # List of (x,y) coordinates in grid units
        self.height = 0  # Height in feet
        self.is_drawing = False

    def add_corner(self, x, y):
        """Add a corner point and draw walls to previous corner"""
        self.corners.append((x, y))
        
        # If we have at least 2 corners, draw wall between them
        if len(self.corners) >= 2:
            x1, y1 = self.corners[-2]
            x2, y2 = self.corners[-1]
            self._draw_wall_line(x1, y1, x2, y2)
        
        print(f"Corner added at grid position ({x}, {y})")

    def _draw_wall_line(self, x1, y1, x2, y2):
        """Draw a line of wall pixels between two points using Bresenham's algorithm"""
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        sx = 1 if x2 > x1 else -1
        sy = 1 if y2 > y1 else -1
        err = dx - dy

        while True:
            if 0 <= x < nx and 0 <= y < ny:
                walls[x, y] = WallType.REFLECTIVE
            
            if x == x2 and y == y2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def complete_room(self):
        """Finalize room by connecting last corner to first"""
        if len(self.corners) < 3:
            print("Need at least 3 corners to complete room")
            return False
            
        # Draw final wall to close the room
        x1, y1 = self.corners[-1]
        x2, y2 = self.corners[0]
        self._draw_wall_line(x1, y1, x2, y2)
        
        self.is_drawing = False
        print(f"Room completed with {len(self.corners)} corners")
        return True

    def clear(self):
        """Clear all corners and walls"""
        self.corners = []
        self.is_drawing = False
        walls.fill(WallType.NONE)

# Replace walls array with Room instance
room = Room()

# Add to the initial state variables (after the other initialization variables)
room_drawing_mode = False  # New variable to track room drawing mode

def convert_to_meters(coords_feet):
    """Convert coordinates from feet to meters"""
    if isinstance(coords_feet, (list, tuple)):
        return [coord * FEET_TO_METERS for coord in coords_feet]
    return coords_feet * FEET_TO_METERS

def calculate_acoustics():
    """Calculate room acoustics using pyroomacoustics"""
    global current_rt60

    if not room.corners or not microphones or not sources:
        print("Need at least one room, one microphone, and one source")
        return

    # Convert corners to meters and prepare for pyroomacoustics
    corners_meters = np.array([convert_to_meters([x * ROOM_WIDTH/nx, y * ROOM_HEIGHT/ny]) for x, y in room.corners]).T
    height_meters = convert_to_meters(room_height)

    # Print debug information
    print(f"Room corners in meters: {corners_meters}")
    
    # Setup pyroomacoustics room
    fs = 48000  # Sampling rate (48 kHz)
    pra_room = pra.Room.from_corners(corners_meters, fs=fs, max_order=5, 
                                    materials=wall_material, ray_tracing=True, 
                                    air_absorption=True)
    
    # Extrude 2D to 3D
    pra_room.extrude(height_meters, materials=ceiling_material)
    
    # Add sources and microphones (convert coordinates to meters)
    for source in sources:
        source_pos = convert_to_meters([
            source.x * ROOM_WIDTH/nx,  # Convert grid x to meters
            source.y * ROOM_HEIGHT/ny,  # Convert grid y to meters
            height_meters/2            # Place at mid-height
        ])
        print(f"Adding source at position (meters): {source_pos}")
        pra_room.add_source(source_pos)
    
<<<<<<< HEAD
    # Create a microphone array from all microphone positions
    mic_positions = np.array([
        convert_to_meters([
            mic.x * ROOM_WIDTH/nx,    # Convert grid x to meters
            mic.y * ROOM_HEIGHT/ny,   # Convert grid y to meters
            height_meters/2           # Place at mid-height
        ]) for mic in microphones
    ]).T  # Transpose to get shape (3, n_mics)
    
    # Add microphone array to the room
    pra_room.add_microphone_array(mic_positions)
    print(f"Added {len(microphones)} microphones to the room simulation")
=======
    for mic in microphones:
        mic_pos = convert_to_meters([
            mic.x * ROOM_WIDTH/nx,    # Convert grid x to meters
            mic.y * ROOM_HEIGHT/ny,   # Convert grid y to meters
            height_meters/2           # Place at mid-height
        ])
        print(f"Adding microphone at position (meters): {mic_pos}")
        pra_room.add_microphone(mic_pos)
>>>>>>> 03cd12879a6de7f74b97c478919f03b0f3719860

    # Setup ray tracing
    pra_room.set_ray_tracing(receiver_radius=0.1, n_rays=10000, energy_thres=1e-7)
    
    # Compute image sources
    pra_room.image_source_model()
    
<<<<<<< HEAD
    # Close any existing figures first
    plt.close('all')
    
    # Create and activate the RIR figure
    fig_rir = plt.figure('Room Impulse Response', figsize=(20, 10))
    plt.clf()  # Clear the figure
    pra_room.plot_rir()  # Plot the room impulse response
    plt.gcf().set_size_inches(20, 10)  # Set figure size after plotting
=======
    # Create separate figures for RIR and frequency response
    plt.figure('Room Impulse Response')
    pra_room.plot_rir()
    rir_fig = plt.gcf()
    rir_fig.set_size_inches(20, 10)
>>>>>>> 03cd12879a6de7f74b97c478919f03b0f3719860
    
    # Calculate RT60
    t60 = pra.experimental.measure_rt60(pra_room.rir[0][0], fs=pra_room.fs, plot=False)
    print(f"The RT60 is {t60 * 1000:.0f} ms")
    
    # Update simulation's RT60
    current_rt60 = t60
    print(f"Updated simulation decay rate based on RT60: {calculate_decay_rate(t60):.6f}")
    
    # Calculate Frequency Response
    chunk_size = 512
    step_size = chunk_size // 2
    min_freq = 20
    max_freq = 2000
    
    rir = pra_room.rir[0][0]
    rir = np.pad(rir, (step_size, len(rir) % chunk_size))
    
    avg_freq_response = np.zeros(chunk_size, dtype=np.complex128)
<<<<<<< HEAD

=======
    
>>>>>>> 03cd12879a6de7f74b97c478919f03b0f3719860
    for i, start_idx in enumerate(range(0, len(rir) - chunk_size, step_size)):
        end_idx = start_idx + chunk_size
        chunk = rir[start_idx:end_idx]
        chunk *= hamming(chunk_size)
        freq_response_chunk = np.fft.fft(chunk)
        avg_freq_response += freq_response_chunk
<<<<<<< HEAD

=======
    
>>>>>>> 03cd12879a6de7f74b97c478919f03b0f3719860
    freq_axis = np.fft.fftfreq(len(avg_freq_response), d=1/pra_room.fs)
    valid_freqs = np.logical_and(min_freq < freq_axis, freq_axis < max_freq)
    freq_axis = freq_axis[valid_freqs]
    freq_response = avg_freq_response[valid_freqs]
<<<<<<< HEAD
=======
    
>>>>>>> 03cd12879a6de7f74b97c478919f03b0f3719860
    freq_response = 20 * np.log10(np.abs(freq_response))
    
    print('Frequency Analysis:')
    print('Num Frequency Bins:', len(freq_axis))
    print('Standard Deviation:', freq_response.std())
    print('Min:', freq_response.min())
    print('Max:', freq_response.max())
    print('Delta:', freq_response.max() - freq_response.min())
    
<<<<<<< HEAD
    # Calculate additional acoustic parameters
    edt = pra.experimental.measure_rt60(pra_room.rir[0][0], fs=pra_room.fs, 
                                      decay_db=10, plot=False) * 6
    print(f"\nEarly Decay Time: {edt * 1000:.0f} ms")
    
    def calculate_clarity(rir, fs, t):
        n = int(t * fs)
        early = np.sum(rir[:n]**2)
        late = np.sum(rir[n:]**2)
        return 10 * np.log10(early / late) if late != 0 else float('inf')
    
    c50 = calculate_clarity(rir, fs, 0.05)
    c80 = calculate_clarity(rir, fs, 0.08)
    print(f"C50 (Speech Clarity): {c50:.1f} dB")
    print(f"C80 (Music Clarity): {c80:.1f} dB")
    
    # Calculate Mean Free Path
    room_volume = pra_room.volume
    # Handle both method and property cases for wall area
    room_surface_area = sum(wall.area() if callable(getattr(wall, 'area', None)) else wall.area for wall in pra_room.walls)
    mean_free_path = 4 * room_volume / room_surface_area
    print(f"Mean Free Path: {mean_free_path:.2f} m")
    
    # Calculate Critical Distance
    avg_absorption = np.mean([wall.absorption for wall in pra_room.walls])
    room_constant = room_surface_area * avg_absorption / (1 - avg_absorption)
    critical_distance = 0.141 * np.sqrt(room_constant)
    print(f"Critical Distance: {critical_distance:.2f} m")
    
    # Create frequency response plot
    fig_freq = plt.figure('Frequency Response')
    plt.clf()
    ax = fig_freq.add_subplot(111)
    ax.plot(freq_axis, freq_response)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('Frequency Response of the Room')
    ax.grid(True)
    
    plt.show(block=False)

def visualize_room_layout():
    """Create a visualization of the room layout with matplotlib"""
    if not room.corners:
        print("No room layout to visualize")
        return
    
    # Close any existing 'Room Layout' figure
    for fig in plt.get_fignums():
        if plt.figure(fig).get_label() == 'Room Layout':
            plt.close(fig)
    
    # Create new figure
    fig = plt.figure('Room Layout', figsize=(10, 10))
    plt.clf()
    ax = fig.add_subplot(111)
    
    # Plot room corners and walls
    corners = np.array(room.corners + [room.corners[0]])  # Add first corner again to close the polygon
    x_coords = [x * ROOM_WIDTH/nx for x, _ in corners]
    y_coords = [y * ROOM_HEIGHT/ny for _, y in corners]
    ax.plot(x_coords, y_coords, 'g-', linewidth=2, label='Walls')
    ax.plot([x * ROOM_WIDTH/nx for x, _ in room.corners],
            [y * ROOM_HEIGHT/ny for _, y in room.corners],
            'ro', label='Corners')
    
    # Plot sources
    for i, source in enumerate(sources):
        x = source.x * ROOM_WIDTH/nx
        y = source.y * ROOM_HEIGHT/ny
        ax.plot(x, y, 'y*', markersize=15, label=f'Source {i+1}')
    
    # Plot microphones
    for i, mic in enumerate(microphones):
        x = mic.x * ROOM_WIDTH/nx
        y = mic.y * ROOM_HEIGHT/ny
        ax.plot(x, y, 'mp', markersize=10, label=f'Mic {i+1}')
    
    # Set labels and title
    ax.set_xlabel('Width (m)')
    ax.set_ylabel('Height (m)')
    ax.set_title('Room Layout')
    ax.grid(True)
    
    # Add legend without duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    # Set equal aspect ratio and display
    ax.set_aspect('equal')
    plt.show(block=False)

def visualize_room_3d():
    """Create a 3D visualization of the room layout with matplotlib"""
    if not room.corners:
        print("No room layout to visualize")
        return
    
    # Close any existing '3D Room Layout' figure
    for fig in plt.get_fignums():
        if plt.figure(fig).get_label() == '3D Room Layout':
            plt.close(fig)
    
    # Create new figure with 3D projection
    fig = plt.figure('3D Room Layout', figsize=(10, 10))
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert room height to meters
    height_meters = room_height * FEET_TO_METERS
    
    # Plot room corners and walls (bottom)
    corners = np.array(room.corners + [room.corners[0]])  # Add first corner again to close the polygon
    x_coords = [x * ROOM_WIDTH/nx for x, _ in corners]
    y_coords = [y * ROOM_HEIGHT/ny for _, y in corners]
    z_coords = [0] * len(corners)  # Bottom of the room
    
    # Plot bottom edges
    ax.plot(x_coords, y_coords, z_coords, 'g-', linewidth=2, label='Bottom Edges')
    
    # Plot top edges
    ax.plot(x_coords, y_coords, [height_meters] * len(corners), 'g-', linewidth=2, label='Top Edges')
    
    # Plot vertical edges
    for i in range(len(room.corners)):
        x = room.corners[i][0] * ROOM_WIDTH/nx
        y = room.corners[i][1] * ROOM_HEIGHT/ny
        ax.plot([x, x], [y, y], [0, height_meters], 'g-', linewidth=2)
    
    # Plot sources with vertical guide lines
    for i, source in enumerate(sources):
        x = source.x * ROOM_WIDTH/nx
        y = source.y * ROOM_HEIGHT/ny
        z = height_meters/2  # Place sources at mid-height
        ax.scatter([x], [y], [z], c='yellow', marker='*', s=200, label=f'Source {i+1}')
        # Add vertical guide line
        ax.plot([x, x], [y, y], [0, height_meters], 'y--', alpha=0.3)
    
    # Plot microphones with vertical guide lines
    for i, mic in enumerate(microphones):
        x = mic.x * ROOM_WIDTH/nx
        y = mic.y * ROOM_HEIGHT/ny
        z = height_meters/2  # Place microphones at mid-height
        ax.scatter([x], [y], [z], c='pink', marker='p', s=100, label=f'Mic {i+1}')
        # Add vertical guide line
        ax.plot([x, x], [y, y], [0, height_meters], 'm--', alpha=0.3)
    
    # Set labels and title
    ax.set_xlabel('Width (m)')
    ax.set_ylabel('Depth (m)')
    ax.set_zlabel('Height (m)')
    ax.set_title('3D Room Layout')
    
    # Add legend without duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    # Set equal aspect ratio for all axes
    max_range = np.array([
        max(x_coords) - min(x_coords),
        max(y_coords) - min(y_coords),
        height_meters
    ]).max() / 2.0
    
    mid_x = (max(x_coords) + min(x_coords)) * 0.5
    mid_y = (max(y_coords) + min(y_coords)) * 0.5
    mid_z = height_meters * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(0, height_meters)
    
    # Enable grid
    ax.grid(True)
    
=======
    # Create frequency response plot in a separate window
    plt.figure('Frequency Response')
    plt.plot(freq_axis, freq_response)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Frequency Response of the Room')
    plt.grid(True)
    
    # Show plots in a non-blocking way
>>>>>>> 03cd12879a6de7f74b97c478919f03b0f3719860
    plt.show(block=False)

def draw():
    # Start with a red background
    screen.fill((128, 0, 0))  # Dark red background

    # Draw wave visualization with lower opacity
    s = pygame.Surface((screen_size, screen_size))
    s.set_alpha(64)  # Make wave visualization more transparent
    s.fill((128, 0, 0))  # Match the red background
    
    # Draw walls and wave visualization
    for x in range(nx):
        for y in range(ny):
            if walls[x, y] != WallType.NONE:
                pygame.draw.rect(screen, WALL_COLORS[walls[x, y]], 
<<<<<<< HEAD
                               (x * scale_x, y * scale_y, scale_x, scale_y))
=======
                               (x * scale_x, y * scale_y + bar_height, scale_x, scale_y))
>>>>>>> 03cd12879a6de7f74b97c478919f03b0f3719860
            else:
                intensity = int((wave[x, y] + 1) * 127.5)
                intensity = max(0, min(255, intensity))
                pygame.draw.rect(screen, (intensity, 0, 0), 
<<<<<<< HEAD
                               (x * scale_x, y * scale_y, scale_x, scale_y))
=======
                               (x * scale_x, y * scale_y + bar_height, scale_x, scale_y))
>>>>>>> 03cd12879a6de7f74b97c478919f03b0f3719860

    # Draw room corners and lines
    if len(room.corners) > 0:
        scaled_points = []
        for x, y in room.corners:
            screen_x = int(x * scale_x)
<<<<<<< HEAD
            screen_y = int(y * scale_y)
=======
            screen_y = int(y * scale_y + bar_height)
>>>>>>> 03cd12879a6de7f74b97c478919f03b0f3719860
            scaled_points.append((screen_x, screen_y))
        
        # Draw lines between corners
        if len(scaled_points) > 1:
            if room.is_drawing:
                pygame.draw.lines(screen, (0, 255, 0), False, scaled_points, 3)
            else:
                pygame.draw.polygon(screen, (0, 255, 0), scaled_points, 3)
        
        # Draw corner points
        for point in scaled_points:
            pygame.draw.circle(screen, (255, 0, 0), point, 6)

    # Draw sources
    for i, source in enumerate(sources):
        border_color = (255, 255, 255) if i == selected_source_index else (100, 100, 100)
        pygame.draw.rect(screen, source.color, 
<<<<<<< HEAD
                        (source.x * scale_x, source.y * scale_y, scale_x, scale_y))
        pygame.draw.rect(screen, border_color,
                        (source.x * scale_x, source.y * scale_y, scale_x, scale_y), 1)
=======
                        (source.x * scale_x, source.y * scale_y + bar_height, scale_x, scale_y))
        pygame.draw.rect(screen, border_color,
                        (source.x * scale_x, source.y * scale_y + bar_height, scale_x, scale_y), 1)
>>>>>>> 03cd12879a6de7f74b97c478919f03b0f3719860

    # Draw microphones (after sources, before UI)
    for mic in microphones:
        pygame.draw.rect(screen, mic.color, 
<<<<<<< HEAD
                        (mic.x * scale_x, mic.y * scale_y, scale_x, scale_y))

    # Draw current settings - only if there are sources
    small_font = pygame.font.Font(None, 24)
=======
                        (mic.x * scale_x, mic.y * scale_y + bar_height, scale_x, scale_y))

    # Draw UI elements
    pygame.draw.rect(screen, GRID_COLOR, (0, 0, screen_size, bar_height))

    # Draw mode indicators and controls (simplified)
    small_font = pygame.font.Font(None, 24)
    controls_text = "F:Single Sound  SPACE:All Sources  S:Add Source  M:Room Mode"
    arrow_controls = "↑↓:Frequency  ←→:Amplitude  Enter:Complete Room  Esc:Clear Room"
    text_surface = small_font.render(controls_text, True, INTENSITY_LINE_COLOR)
    arrow_surface = small_font.render(arrow_controls, True, INTENSITY_LINE_COLOR)
    screen.blit(text_surface, (200, 15))
    screen.blit(arrow_surface, (200, 35))

    # Draw current settings - only if there are sources
>>>>>>> 03cd12879a6de7f74b97c478919f03b0f3719860
    if sources:
        source = sources[selected_source_index]
        freq_text = f"Freq: {source.frequency:.1f} Hz"
        amp_text = f"Amp: {source.amplitude:.1f}"
    else:
        freq_text = "Freq: N/A"
        amp_text = "Amp: N/A"
    
    mode_text = "Mode: "
    if room_drawing_mode:
        mode_text += "Room Drawing"
    elif mic_mode:
        mode_text += "Microphone"
<<<<<<< HEAD
    elif height_adjustment_mode:
        mode_text += "Height Adjustment"
=======
>>>>>>> 03cd12879a6de7f74b97c478919f03b0f3719860
    else:
        mode_text += "None"
    
    freq_surface = small_font.render(freq_text, True, INTENSITY_LINE_COLOR)
    amp_surface = small_font.render(amp_text, True, INTENSITY_LINE_COLOR)
    mode_surface = small_font.render(mode_text, True, INTENSITY_LINE_COLOR)
    
    screen.blit(freq_surface, (500, 10))
    screen.blit(amp_surface, (500, 35))
    screen.blit(mode_surface, (650, 10))

<<<<<<< HEAD
    # Draw bottom control panel background
    bottom_panel_rect = pygame.Rect(0, screen_size, screen_size, bottom_panel_height)
    pygame.draw.rect(screen, GRID_COLOR, bottom_panel_rect)
    
    # Draw divider line
    pygame.draw.line(screen, INTENSITY_LINE_COLOR, 
                    (0, screen_size),
                    (screen_size, screen_size), 2)

    # Create sections in bottom panel
    panel_y = screen_size + 10  # Starting Y position for panel content
    left_margin = 20
    
    # Left section: Room Controls
    height_text = f"Room Height: {height_input_text} ft" if height_input_active else f"Room Height: {room_height:.1f} ft ({room_height * FEET_TO_METERS:.1f} m)"
    height_surface = small_font.render(height_text, True, INTENSITY_LINE_COLOR)
    screen.blit(height_surface, (left_margin, panel_y))
    
    # Room controls help text
    room_controls = ["H: Height Mode", "M: Room Mode", "X: Mic Mode"]
    for i, control in enumerate(room_controls):
        room_surface = small_font.render(control, True, INTENSITY_LINE_COLOR)
        screen.blit(room_surface, (left_margin, panel_y + 25 + i*20))
    
    # Middle section: Source Controls
    source_text = "Source Controls:"
    source_controls = ["F: Single Source", "SPACE: All Sources", "S: Add Source"]
    source_surface = small_font.render(source_text, True, INTENSITY_LINE_COLOR)
    screen.blit(source_surface, (screen_size//3, panel_y))
    for i, control in enumerate(source_controls):
        control_surface = small_font.render(control, True, INTENSITY_LINE_COLOR)
        screen.blit(control_surface, (screen_size//3, panel_y + 25 + i*20))
    
    # Right section: Analysis Controls
    analysis_text = "Analysis:"
    analysis_controls = ["C: Calculate", "V: 2D View", "3: 3D View"]
    analysis_surface = small_font.render(analysis_text, True, INTENSITY_LINE_COLOR)
    screen.blit(analysis_surface, (2*screen_size//3, panel_y))
    for i, control in enumerate(analysis_controls):
        control_surface = small_font.render(control, True, INTENSITY_LINE_COLOR)
        screen.blit(control_surface, (2*screen_size//3, panel_y + 25 + i*20))

    # Additional status info in the bottom row
    status_y = panel_y + 50
    if height_adjustment_mode:
        status_text = "HEIGHT ADJUSTMENT MODE - Use Up/Down arrows"
    elif room_drawing_mode:
        status_text = "ROOM DRAWING MODE - Click to place corners"
    elif mic_mode:
        status_text = "MICROPHONE PLACEMENT MODE - Click to place mics"
    else:
        status_text = "Press ESC to clear room and start over"
    
    status_surface = small_font.render(status_text, True, INTENSITY_LINE_COLOR)
    status_rect = status_surface.get_rect(center=(screen_size//2, status_y + 10))
    screen.blit(status_surface, status_rect)
=======
    # Add height input field at the bottom
    height_text = f"Room Height (ft): {height_input_text}" if height_input_active else f"Room Height (ft): {room_height}"
    height_surface = small_font.render(height_text, True, INTENSITY_LINE_COLOR)
    screen.blit(height_surface, (10, screen_size + 30))  # Position below the simulation area

    # Add calculation instruction
    calc_text = "Press C to calculate acoustics"
    calc_surface = small_font.render(calc_text, True, INTENSITY_LINE_COLOR)
    screen.blit(calc_surface, (screen_size - 200, screen_size + 30))
>>>>>>> 03cd12879a6de7f74b97c478919f03b0f3719860

    pygame.display.flip()

def erase_at_position(x: int, y: int):
    global sources, selected_source_index
    # First try to erase any sources at this position
    for i, source in enumerate(reversed(sources)):
        if source.x == x and source.y == y:
            if len(sources) > 1:  # Keep at least one source
                sources.pop(len(sources) - 1 - i)
                selected_source_index = min(selected_source_index, len(sources) - 1)
                return True
    # If no source was erased, erase walls
    if 0 <= x < nx and 0 <= y < ny:
        walls[x, y] = WallType.NONE
    return False

def reset_simulation():
    global wave, wave_prev, walls, sources, selected_source_index
    # Reset wave fields
    wave.fill(0)
    wave_prev.fill(0)
    # Reset walls
    walls.fill(WallType.NONE)
    # Reset sources to single initial source
    sources = [SoundSource(nx // 6, ny // 6, frequency=440.0, amplitude=1.0, color=SOURCE_COLORS[0])]
    selected_source_index = 0

def place_sound_source():
    global sources, selected_source_index
    mouse_x, mouse_y = pygame.mouse.get_pos()
<<<<<<< HEAD
    if mouse_y >= screen_size:  # Don't place sources in the bottom panel
        return
    grid_x = int(mouse_x // scale_x)
    grid_y = int(mouse_y // scale_y)
=======
    if mouse_y < bar_height:
        return
    grid_x = int(mouse_x // scale_x)
    grid_y = int((mouse_y - bar_height) // scale_y)
>>>>>>> 03cd12879a6de7f74b97c478919f03b0f3719860

    if 0 <= grid_x < nx and 0 <= grid_y < ny and walls[grid_x, grid_y] == WallType.NONE:
        # Create new source with cycling colors
        new_source = SoundSource(
            x=grid_x, 
            y=grid_y,
            frequency=440.0,
            amplitude=1.0,
            color=SOURCE_COLORS[len(sources) % len(SOURCE_COLORS)]
        )
        sources.append(new_source)
        selected_source_index = len(sources) - 1

# Main loop
running = True
<<<<<<< HEAD
height_adjustment_mode = False  # Track if we're adjusting height

=======
>>>>>>> 03cd12879a6de7f74b97c478919f03b0f3719860
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_f:
                sources[selected_source_index].active = True
            elif event.key == pygame.K_SPACE:
                for source in sources:
                    source.active = True
<<<<<<< HEAD
            elif event.key == pygame.K_h:
                height_adjustment_mode = True
            elif event.key == pygame.K_UP:
                if height_adjustment_mode:
                    # Convert step from meters to feet
                    room_height = min(50.0, room_height + HEIGHT_STEP / FEET_TO_METERS)
                    print(f"Room height: {room_height * FEET_TO_METERS:.1f} meters")
                else:
                    sources[selected_source_index].frequency = min(2000.0, sources[selected_source_index].frequency + FREQ_STEP)
            elif event.key == pygame.K_DOWN:
                if height_adjustment_mode:
                    # Convert step from meters to feet
                    room_height = max(2.0, room_height - HEIGHT_STEP / FEET_TO_METERS)
                    print(f"Room height: {room_height * FEET_TO_METERS:.1f} meters")
                else:
                    sources[selected_source_index].frequency = max(20.0, sources[selected_source_index].frequency - FREQ_STEP)
=======
            elif event.key == pygame.K_UP:
                sources[selected_source_index].frequency = min(2000.0, sources[selected_source_index].frequency + FREQ_STEP)
            elif event.key == pygame.K_DOWN:
                sources[selected_source_index].frequency = max(20.0, sources[selected_source_index].frequency - FREQ_STEP)
>>>>>>> 03cd12879a6de7f74b97c478919f03b0f3719860
            elif event.key == pygame.K_RIGHT:
                sources[selected_source_index].amplitude = min(2.0, sources[selected_source_index].amplitude + AMP_STEP)
            elif event.key == pygame.K_LEFT:
                sources[selected_source_index].amplitude = max(0.1, sources[selected_source_index].amplitude - AMP_STEP)
            elif event.key == pygame.K_s:
                place_sound_source()
            elif event.key == pygame.K_TAB:
                selected_source_index = (selected_source_index + 1) % len(sources)
            elif event.key == pygame.K_DELETE:
                if len(sources) > 1:
                    sources.pop(selected_source_index)
                    selected_source_index = min(selected_source_index, len(sources) - 1)
            elif event.key == pygame.K_RETURN:
                if room.complete_room():
                    room.is_drawing = False
            elif event.key == pygame.K_ESCAPE:
                room.clear()
                room.is_drawing = False
                sources.clear()  # Clear all sound sources
                microphones.clear()  # Clear all microphones
                print("Room, sound sources, and microphones cleared")
            elif event.key == pygame.K_m:
                room_drawing_mode = not room_drawing_mode
                if room_drawing_mode:
                    print("Room drawing mode activated - click to place corners")
                else:
                    print("Room drawing mode deactivated")
            elif event.key == pygame.K_x:
                mic_mode = not mic_mode
                room_drawing_mode = False  # Disable room drawing mode when entering mic mode
                if mic_mode:
                    print("Microphone placement mode activated")
                else:
                    print("Microphone placement mode deactivated")
            elif event.key == pygame.K_h:
                height_input_active = not height_input_active
                if height_input_active:
                    height_input_text = str(room_height)
                    print("Room height input activated")
                else:
                    try:
                        new_height = float(height_input_text)
                        if new_height > 0:
                            room_height = new_height
                        print(f"Room height set to {room_height} feet")
                    except ValueError:
                        print("Invalid height value")
                    height_input_active = False
            elif event.key == pygame.K_c:
                if len(room.corners) >= 3 and len(microphones) > 0 and len(sources) > 0:
                    calculate_acoustics()
                else:
                    print("Need a complete room, at least one microphone, and one source to calculate acoustics")
<<<<<<< HEAD
            elif event.key == pygame.K_v:
                visualize_room_layout()
            elif event.key == pygame.K_3:  # Press '3' for 3D view
                visualize_room_3d()
=======
>>>>>>> 03cd12879a6de7f74b97c478919f03b0f3719860
            elif height_input_active:
                if event.key == pygame.K_RETURN:
                    try:
                        new_height = float(height_input_text)
                        if new_height > 0:
                            room_height = new_height
                            print(f"Room height set to {new_height} feet")
                    except ValueError:
                        print("Invalid height value")
                    height_input_active = False
                elif event.key == pygame.K_BACKSPACE:
                    height_input_text = height_input_text[:-1]
                elif event.unicode.isnumeric() or event.unicode == '.':
                    height_input_text += event.unicode
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_f:
                sources[selected_source_index].active = False
<<<<<<< HEAD
            elif event.key == pygame.K_h:
                height_adjustment_mode = False
=======
>>>>>>> 03cd12879a6de7f74b97c478919f03b0f3719860
            elif event.key == pygame.K_SPACE:
                for source in sources:
                    source.active = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_x, mouse_y = pygame.mouse.get_pos()
<<<<<<< HEAD
                if mouse_y < screen_size:  # Only handle clicks in the main area
                    grid_x = int(mouse_x // scale_x)
                    grid_y = int(mouse_y // scale_y)
=======
                if mouse_y >= bar_height:
                    grid_x = int(mouse_x // scale_x)
                    grid_y = int((mouse_y - bar_height) // scale_y)
>>>>>>> 03cd12879a6de7f74b97c478919f03b0f3719860
                    if 0 <= grid_x < nx and 0 <= grid_y < ny:
                        if room_drawing_mode:
                            room.add_corner(grid_x, grid_y)
                            room.is_drawing = True
                        elif mic_mode:
                            microphones.append(Microphone(grid_x, grid_y))
                            print(f"Microphone placed at grid position ({grid_x}, {grid_y})")

    update_wave()
    draw()
    clock.tick(60)

pygame.quit()
sys.exit()
