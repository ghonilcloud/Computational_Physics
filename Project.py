import numpy as np
import pygame
import sys
from dataclasses import dataclass
from typing import List
import colorsys
from math import floor
import pyroomacoustics as pra
from numpy import hamming
import threading  # For running plots in separate threads
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid thread issues
import matplotlib.pyplot as plt

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

# Simple dropdown menu class for material selection
class DropdownMenu:
    def __init__(self, x, y, width, height, options, label="Dropdown", dropup=True):
        self.rect = pygame.Rect(x, y, width, height)
        self.options = options  # List of (name, value) tuples
        self.expanded = False
        self.selected_index = 0  # Default to first option
        self.label = label
        self.option_height = 30  # Height of each dropdown option
        self.font = pygame.font.Font(None, 24)
        self.active = False  # To track if this dropdown is selected
        self.dropup = dropup  # Whether to show options above (True) or below (False) the button

    def draw(self, screen):
        # Draw the main dropdown button
        if self.active:
            button_color = (180, 180, 220)  # Light blue when active
        else:
            button_color = (80, 80, 100)  # Darker when inactive

        pygame.draw.rect(screen, button_color, self.rect)
        pygame.draw.rect(screen, (255, 255, 255), self.rect, 2)  # White border
        
        # Draw the label and selected option
        label_text = self.font.render(f"{self.label}: {self.options[self.selected_index][0]}", True, (255, 255, 255))
        screen.blit(label_text, (self.rect.x + 10, self.rect.y + (self.rect.height - label_text.get_height()) // 2))
        
        arrow_points = [
            (self.rect.right - 20, self.rect.centery - 5),
            (self.rect.right - 30, self.rect.centery + 5),
            (self.rect.right - 10, self.rect.centery + 5)
        ]

        pygame.draw.polygon(screen, (255, 255, 255), arrow_points)
        
        # If expanded, draw options list (either above or below button)
        if self.expanded:
            total_height = len(self.options) * self.option_height
            
            for i, (option_name, _) in enumerate(self.options):
                if self.dropup:
                    # Position above
                    y_pos = self.rect.y - total_height + i * self.option_height
                else:
                    # Position below
                    y_pos = self.rect.y + self.rect.height + i * self.option_height
                
                option_rect = pygame.Rect(
                    self.rect.x,
                    y_pos,
                    self.rect.width, 
                    self.option_height
                )
                
                # Highlight selected option
                if i == self.selected_index:
                    pygame.draw.rect(screen, (100, 100, 160), option_rect)
                else:
                    pygame.draw.rect(screen, (60, 60, 80), option_rect)
                    
                pygame.draw.rect(screen, (255, 255, 255), option_rect, 1)  # White border
                
                # Draw option text
                option_text = self.font.render(option_name, True, (255, 255, 255))
                screen.blit(option_text, (option_rect.x + 10, option_rect.y + (option_rect.height - option_text.get_height()) // 2))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Check if the main dropdown button was clicked
            if self.rect.collidepoint(event.pos):
                self.expanded = not self.expanded
                return True
            
            # If expanded, check if an option was clicked
            if self.expanded:
                total_height = len(self.options) * self.option_height
                
                for i, _ in enumerate(self.options):
                    if self.dropup:
                        # Position above
                        y_pos = self.rect.y - total_height + i * self.option_height
                    else:
                        # Position below
                        y_pos = self.rect.y + self.rect.height + i * self.option_height
                    
                    option_rect = pygame.Rect(
                        self.rect.x,
                        y_pos,
                        self.rect.width, 
                        self.option_height
                    )
                    
                    if option_rect.collidepoint(event.pos):
                        self.selected_index = i
                        self.expanded = False
                        return True
                        
            # Click elsewhere closes the dropdown
            if self.expanded:
                self.expanded = False
                return True
        
        return False
    
    def get_selected_value(self):
        """Return the selected material object"""
        return self.options[self.selected_index][1]    
    
    def set_active(self, active):
        """Set this dropdown as active (or not)"""
        self.active = active
          
    def get_available_options_text(self):
        """Return a formatted string of available materials for this dropdown"""
        if len(self.options) <= 4:
            # For smaller lists, show all options separated by commas
            options_text = ", ".join([f"{i+1}:{opt[0]}" for i, opt in enumerate(self.options)])
            return f"{self.label} options: {options_text}"
        else:
            # For longer lists, show first few options and "more..."
            options_text = ", ".join([f"{i+1}:{opt[0]}" for i, opt in enumerate(self.options[:3])])
            return f"{self.label} options: {options_text}, ... (Press TAB and number keys to select)"

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
ROOM_WIDTH = 15.0
ROOM_HEIGHT = 15.0

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

# Define wall constants
WALL_NONE = 0.0
WALL_PRESENT = 1.0

# Modify walls array to store wall presence (0.0 for no wall, 1.0 for wall)
walls = np.zeros((nx, ny), dtype=float)

# Pygame initialization
pygame.init()
screen_width = 900  # Width of the window
screen_height = 700  # Height of the main visualization area
bottom_panel_height = 150  # Increased height for bottom control panel with dropdowns
screen = pygame.display.set_mode((screen_width, screen_height + bottom_panel_height))
pygame.display.set_caption('Sound Wave Propagation')
clock = pygame.time.Clock()

# Colors
WALL_COLOR = (128, 0, 0)  # Dark red background/wall color

SOURCE_COLORS = [
    (255, 255, 0),  # Yellow
    (0, 255, 255),  # Cyan
    (255, 0, 255),  # Magenta
    (255, 128, 0)   # Orange
]

MEASUREMENT_COLOR = (255, 0, 255)
GRID_COLOR = (40, 40, 40)  # Darker gray for better contrast
INTENSITY_LINE_COLOR = (220, 220, 220)  # Lighter gray for text and lines

# Scaling factor
scale_x = screen_width / nx
scale_y = screen_height / ny

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

# Available frequencies (Hz)
AVAILABLE_FREQUENCIES = [125, 250, 500, 1000, 2000, 4000, 8000]  # Standard acoustic frequencies
DEFAULT_FREQUENCY_INDEX = 2  # Start with 500 Hz (index 2)

# Amplitude step size
AMP_STEP = 0.2    # Amplitude change per keypress

# Add these constants after the other room/grid parameters
TEST_BOX_SIZE = 20  # Size of test boxes in grid units
TEST_BOX_SPACING = 10  # Space between boxes

FEET_TO_METERS = 0.3048  # Conversion factor
HEIGHT_STEP = 0.3  # Height change in meters per keypress
DEFAULT_ROOM_HEIGHT = 3.0 / FEET_TO_METERS  # Default height in feet (3 meters)

# Function to get reflection/absorption coefficient from wall material
def get_wall_coefficient(material):
    """
    Extract reflection coefficient from wall material for 2D visualization.
    Higher absorption = lower reflection coefficient.
    Returns a value between 0.0 (complete absorption) and 1.0 (complete reflection).
    """
    # Extract average absorption value across frequencies
    if hasattr(material, 'absorption_coeffs'):
        # Check if absorption_coeffs is a dictionary or a list
        if isinstance(material.absorption_coeffs, dict):
            # If it's a dictionary, get values and calculate mean
            absorption = np.mean(list(material.absorption_coeffs.values()))
        else:
            # If it's a list or another iterable, calculate mean directly
            absorption = np.mean(material.absorption_coeffs)
    elif hasattr(material, 'energy_absorption'):
        # Direct absorption value
        absorption = material.energy_absorption
    else:
        # Default moderate absorption if can't determine
        absorption = 0.3
    
    # Convert absorption to reflection (1 - absorption)
    # Higher absorption = lower reflection
    return 1.0 - absorption

#List of all materials
# total_reflection = pra.Material(energy_absorption=0, scattering=0.25)
# total_absorption = pra.Material(energy_absorption=1, scattering=0.25)

# Wall materials
unpainted_concrete = pra.Material('unpainted_concrete')
brickwork = pra.Material('brickwork')
brick_wall_rough = pra.Material('brick_wall_rough')
rough_concrete = pra.Material('rough_concrete')
limestone_wall = pra.Material('limestone_wall')
glass_3mm = pra.Material('glass_3mm')
wooden_lining = pra.Material('wooden_lining')

# Ceiling materials
wooden_lining = pra.Material('wooden_lining')
ceiling_plasterboard = pra.Material('ceiling_plasterboard')
unpainted_concrete = pra.Material('unpainted_concrete')
ceiling_fissured_tile = pra.Material('ceiling_fissured_tile')
ceiling_metal_panel = pra.Material('ceiling_metal_panel')

# Floor materials
ceramic_tiles = pra.Material('ceramic_tiles')
concrete_floor = pra.Material('concrete_floor')
marble_floor = pra.Material('marble_floor')
carpet_hairy = pra.Material('carpet_hairy')
carpet_thin = pra.Material('carpet_thin')
linoleum_on_concrete = pra.Material('linoleum_on_concrete')

# Define material options for each surface type
# Wall materials
wall_material_options = [
    ("Unpainted Concrete", unpainted_concrete),
    ("Brickwork", brickwork),
    ("Brick Wall (Rough)", brick_wall_rough),
    ("Rough Concrete", rough_concrete),
    ("Limestone Wall", limestone_wall),
    ("Glass (3mm)", glass_3mm),
    ("Wooden Lining", wooden_lining),
]

# Ceiling materials
ceiling_material_options = [
    ("Wooden Lining", wooden_lining),
    ("Plasterboard Ceiling", ceiling_plasterboard),
    ("Unpainted Concrete", unpainted_concrete),
    ("Fissured Acoustic Tile", ceiling_fissured_tile),
    ("Metal Panel Ceiling", ceiling_metal_panel),
]

# Floor materials
floor_material_options = [
    ("Ceramic Tiles", ceramic_tiles),
    ("Concrete Floor", concrete_floor),
    ("Marble Floor", marble_floor),
    ("Hairy Carpet", carpet_hairy),
    ("Thin Carpet", carpet_thin),
    ("Linoleum on Concrete", linoleum_on_concrete),
]

# Function to get material from dropdown (no frequency restrictions)
def get_material_for_frequency(dropdown, frequency):
    """
    Returns the material selected in the dropdown without any frequency-based restrictions.
    This allows complete freedom in material selection regardless of the frequency.
    """
    # Simply return the selected material from the dropdown
    return dropdown.get_selected_value()

# Add acoustic materials
#These lines define acoustic materials for the walls, ceiling, and floor of the room using the 
# pyroomacoustics library. Each material is characterized by energy absorption and scattering.
wall_dropdown = DropdownMenu(20, screen_height + 10, 260, 30, wall_material_options, "Wall")
ceiling_dropdown = DropdownMenu(300, screen_height + 10, 260, 30, ceiling_material_options, "Ceiling") 
floor_dropdown = DropdownMenu(580, screen_height + 10, 260, 30, floor_material_options, "Floor")

# Set default selection (can be adjusted as needed)
wall_dropdown.selected_index = 0  # Unpainted Concrete
ceiling_dropdown.selected_index = 0  # Wooden Lining
floor_dropdown.selected_index = 0  # Ceramic Tiles

# Active dropdown (for keyboard navigation)
active_dropdown_index = 0
dropdowns = [wall_dropdown, ceiling_dropdown, floor_dropdown]
dropdowns[active_dropdown_index].set_active(True)

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

def update_wave():
    global wave, wave_prev
    wave_next = np.copy(wave)

    # Get the wall material reflection coefficient
    if len(sources) > 0:
        current_frequency = sources[selected_source_index].frequency
    else:
        current_frequency = 500  # Default if no sources
    
    # Get the wall material from dropdown
    wall_material = get_material_for_frequency(wall_dropdown, current_frequency)
    
    # Calculate reflection coefficient for walls (1.0 = perfect reflection, 0.0 = complete absorption)
    wall_reflection_coeff = get_wall_coefficient(wall_material)

    # Calculate current decay rate based on RT60
    current_decay = calculate_decay_rate(current_rt60)    # Iterates through each sound source, and only generates noise if source is active.
    for source in sources:
        if source.active:  # Check individual source activation
            t = pygame.time.get_ticks() / 1000.0
            
            # Calculate frequency-dependent enhancement factor
            # This ensures higher frequencies have enough energy to propagate
            # through the simulation despite the grid resolution limitations
            freq_enhancement = min(10, max(1, source.frequency / 250))
            
            # Generate multiple wave samples for higher frequencies to avoid temporal aliasing
            # Higher frequencies need more samples per frame for proper representation
            num_samples = max(1, int(source.frequency / 30))
            wave_contribution = 0
            
            for i in range(num_samples):
                # Calculate time offsets within this frame for better sampling
                sample_time = t + (i * (1.0/60.0) / max(1, num_samples))
                wave_contribution += np.sin(2 * np.pi * source.frequency * sample_time)
            
            # Add the averaged wave contribution with frequency enhancement
            wave[source.x, source.y] += source.amplitude * freq_enhancement * (wave_contribution / num_samples)
            
            # Source.amplitude is the wave amplitude/height/intensity
            # The wave frequency is determined by source.frequency
            # freq_enhancement increases the amplitude for higher frequencies
            # to compensate for grid resolution limitations
    
        for x in range(1, nx - 1):
            for y in range(1, ny - 1):
                # Check if this cell is a wall
                is_wall = walls[x, y] > 0
                if not is_wall:
                    # Get current frequency for adaptive wave propagation
                    current_freq = sources[selected_source_index].frequency if sources and selected_source_index < len(sources) else 500
                    
                    # For high frequencies, adjust propagation with a small damping factor
                    # to compensate for numerical dispersion that affects high frequencies
                    damping_factor = max(0.985, 1.0 - (current_freq / 10000))
                    
                    # Enhanced wave equation with frequency-aware damping
                    wave_next[x, y] = damping_factor * (2 * wave[x, y] - wave_prev[x, y] +
                                    # c is speed of sound, dt is time step, dx is spatial step
                                    (c * dt / dx) ** 2 *
                                    # Gets the horizontal and vertical neighbors
                                    (wave[x + 1, y] + wave[x - 1, y] +
                                    wave[x, y + 1] + wave[x, y - 1] - 
                                    4 * wave[x, y]))
                else:
                    # Apply reflection/absorption based on the wall material
                    incident_wave = wave[x, y]
                    wave_next[x, y] = wall_reflection_coeff * incident_wave

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
                walls[x, y] = WALL_PRESENT
            
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
        walls.fill(WALL_NONE)

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
      # Get current sound source frequency for materials
    current_frequency = sources[selected_source_index].frequency if sources else 500
      # Get materials from user selections
    selected_wall_material = get_material_for_frequency(wall_dropdown, current_frequency) 
    selected_ceiling_material = get_material_for_frequency(ceiling_dropdown, current_frequency)
    selected_floor_material = get_material_for_frequency(floor_dropdown, current_frequency)
    
    print(f"Using selected materials (frequency: {current_frequency} Hz):")
    print(f"  Wall: {wall_dropdown.options[wall_dropdown.selected_index][0]}")
    print(f"  Ceiling: {ceiling_dropdown.options[ceiling_dropdown.selected_index][0]}")
    print(f"  Floor: {floor_dropdown.options[floor_dropdown.selected_index][0]}")
    
    # Setup pyroomacoustics room
    fs = 48000  # Sampling rate (48 kHz)
    pra_room = pra.Room.from_corners(corners_meters, fs=fs, max_order=5, 
                                    materials=selected_wall_material, ray_tracing=True, 
                                    air_absorption=True)
    
    # Extrude 2D to 3D (ceiling and floor materials)
    pra_room.extrude(height_meters, materials={
        'ceiling': selected_ceiling_material,
        'floor': selected_floor_material
    })
    
    # Add sources and microphones (convert coordinates to meters)
    for source in sources:
        source_pos = convert_to_meters([
            source.x * ROOM_WIDTH/nx,  # Convert grid x to meters
            source.y * ROOM_HEIGHT/ny,  # Convert grid y to meters
            height_meters/2            # Place at mid-height
        ])
        print(f"Adding source at position (meters): {source_pos}")
        pra_room.add_source(source_pos)
    
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

    # Setup ray tracing
    pra_room.set_ray_tracing(receiver_radius=0.1, n_rays=10000, energy_thres=1e-7)
    
    # Compute image sources
    pra_room.image_source_model()
    
    # Compute the room impulse response (RIR)
    pra_room.compute_rir()
    print("Room Impulse Response computed successfully")
      
    # Close any existing figures first
    plt.close('all')
    
    # Create the RIR plot directly to make sure it's in the right figure
    fig_rir = plt.figure('Room Impulse Response', figsize=(20, 10))
    plt.clf()  # Clear the figure
    
    # Check if RIR was computed successfully
    if pra_room.rir is None:
        # If no RIR is available, show a message
        ax = fig_rir.add_subplot(111)
        ax.text(0.5, 0.5, "No Room Impulse Response data available.\nTry adjusting room parameters.", 
                horizontalalignment='center', verticalalignment='center', fontsize=16)
        ax.set_axis_off()
    else:
        # Manually plot the Room Impulse Response instead of using pra_room.plot_rir()
        for i in range(len(pra_room.rir)):
            for j in range(len(pra_room.rir[i])):
                plt.subplot(len(pra_room.rir), len(pra_room.rir[0]), i*len(pra_room.rir[0])+j+1)
                plt.plot(pra_room.rir[i][j])
                plt.title(f'Source {j} to Mic {i}')
                plt.xlabel('Time (samples)')
                plt.ylabel('Amplitude')
                plt.grid(True)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Make sure we use the current figure
    fig_rir = plt.gcf()
    fig_rir.set_size_inches(20, 10)  # Set figure size after plotting
    
    # Calculate RT60 if RIR is available
    if pra_room.rir is not None and len(pra_room.rir) > 0 and len(pra_room.rir[0]) > 0:
        t60 = pra.experimental.measure_rt60(pra_room.rir[0][0], fs=pra_room.fs, plot=False)
        print(f"The RT60 is {t60 * 1000:.0f} ms")
        
        # Update simulation's RT60
        current_rt60 = t60
        print(f"Updated simulation decay rate based on RT60: {calculate_decay_rate(t60):.6f}")
    else:
        print("Unable to calculate RT60: Room impulse response not available")
        # Keep the current RT60 value
    
    # Calculate Frequency Response if RIR is available
    if pra_room.rir is not None and len(pra_room.rir) > 0 and len(pra_room.rir[0]) > 0:
        chunk_size = 512
        step_size = chunk_size // 2
        min_freq = 20
        max_freq = 2000
        
        rir = pra_room.rir[0][0]
        rir = np.pad(rir, (step_size, len(rir) % chunk_size))
        
        avg_freq_response = np.zeros(chunk_size, dtype=np.complex128)

        for i, start_idx in enumerate(range(0, len(rir) - chunk_size, step_size)):
            end_idx = start_idx + chunk_size
            chunk = rir[start_idx:end_idx]
            chunk *= hamming(chunk_size)
            freq_response_chunk = np.fft.fft(chunk)
            avg_freq_response += freq_response_chunk

        freq_axis = np.fft.fftfreq(len(avg_freq_response), d=1/pra_room.fs)
        valid_freqs = np.logical_and(min_freq < freq_axis, freq_axis < max_freq)
        freq_axis = freq_axis[valid_freqs]
        freq_response = avg_freq_response[valid_freqs]
        freq_response = 20 * np.log10(np.abs(freq_response))
        
        print('Frequency Analysis:')
        print('Num Frequency Bins:', len(freq_axis))
        print('Standard Deviation:', freq_response.std())
        print('Min:', freq_response.min())
        print('Max:', freq_response.max())
        print('Delta:', freq_response.max() - freq_response.min())
    else:
        print("Unable to perform frequency analysis: Room impulse response not available")
        # Create default values for plotting
        freq_axis = np.linspace(20, 2000, 100)  # Default frequency range
        freq_response = np.zeros_like(freq_axis)  # Empty response
    
    # Calculate additional acoustic parameters if RIR is available
    if pra_room.rir is not None and len(pra_room.rir) > 0 and len(pra_room.rir[0]) > 0:
        # Define clarity calculation function
        def calculate_clarity(rir, fs, t):
            n = int(t * fs)
            early = np.sum(rir[:n]**2)
            late = np.sum(rir[n:]**2)
            return 10 * np.log10(early / late) if late != 0 else float('inf')
        
        # Calculate Early Decay Time (EDT)
        edt = pra.experimental.measure_rt60(pra_room.rir[0][0], fs=pra_room.fs, 
                                        decay_db=10, plot=False) * 6
        print(f"\nEarly Decay Time: {edt * 1000:.0f} ms")
        
        # Calculate clarity metrics
        c50 = calculate_clarity(rir, fs, 0.05)
        c80 = calculate_clarity(rir, fs, 0.08)
        print(f"C50 (Speech Clarity): {c50:.1f} dB")
        print(f"C80 (Music Clarity): {c80:.1f} dB")
    else:
        print("Unable to calculate additional acoustic parameters: Room impulse response not available")
    
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
    print(f"Critical Distance: {critical_distance:.2f} m")    # Create frequency response plot
    fig_freq = plt.figure('Frequency Response')
    plt.clf()
    ax = fig_freq.add_subplot(111)
    
    if pra_room.rir is not None and len(pra_room.rir) > 0 and len(pra_room.rir[0]) > 0:
        ax.plot(freq_axis, freq_response)
        ax.set_title('Frequency Response of the Room')
    else:
        ax.text(0.5, 0.5, "No Frequency Response data available.\nTry adjusting room parameters.", 
               horizontalalignment='center', verticalalignment='center', fontsize=16)
        ax.set_title('Frequency Response - No Data')
        
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.grid(True)
    
    # Make sure both figures are finalized before showing them
    plt.figure(fig_rir.number)
    plt.tight_layout()
    
    plt.figure(fig_freq.number)
    plt.tight_layout()
    
    # Show plots in separate threads to avoid affecting the main window
    show_plot_in_thread(fig_rir)
    show_plot_in_thread(fig_freq)

def show_plot_in_thread(fig):
    """Shows a matplotlib figure in a separate thread to avoid impacting the main window"""
    def _show_plot():
        # Save figure to a temporary file and open it with system default viewer
        import os
        import tempfile
        
        # Create a unique filename based on the figure's label or a random ID
        if hasattr(fig, 'get_label') and fig.get_label():
            filename = f"{fig.get_label().replace(' ', '_')}.png"
        else:
            filename = f"plot_{id(fig)}.png"
        
        # Check if the figure has any axes with content
        if len(fig.axes) == 0:
            print(f"Warning: Figure {filename} has no axes! Adding dummy content.")
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", 
                   horizontalalignment='center', verticalalignment='center')
        
        # Save to temp directory
        filepath = os.path.join(tempfile.gettempdir(), filename)
        fig.savefig(filepath, dpi=100)
        print(f"Saved plot to {filepath}")
        
        # Open with default system viewer
        try:
            os.startfile(filepath)  # Windows-specific
        except AttributeError:
            # For non-Windows systems (not relevant here but good practice)
            import subprocess
            subprocess.call(('xdg-open', filepath))  # Linux
        
    thread = threading.Thread(target=_show_plot)
    thread.daemon = True  # Thread will close when main program exits
    thread.start()

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
    show_plot_in_thread(fig)

def visualize_room_3d():
    """Create a 3D visualization of the room layout with matplotlib in a detached process"""
    if not room.corners:
        print("No room layout to visualize")
        return
    
    # Save all necessary data to a temporary file
    import tempfile
    import pickle
    import os
    import subprocess
    import sys
    
    print("Launching 3D visualization in a separate window...")
    
    data = {
        "corners": room.corners,
        "height": room_height * FEET_TO_METERS,
        "room_width": ROOM_WIDTH, 
        "room_height": ROOM_HEIGHT,
        "nx": nx,
        "ny": ny,
        "sources": [(s.x, s.y, s.frequency, s.amplitude, s.color) for s in sources],
        "microphones": [(m.x, m.y) for m in microphones]
    }
    
    temp_dir = tempfile.gettempdir()
    data_file = os.path.join(temp_dir, "room_3d_data.pkl")
    with open(data_file, "wb") as f:
        pickle.dump(data, f)
    
    # Create a script file that will be run as a separate process
    script_file = os.path.join(temp_dir, "show_3d_room.py")
    with open(script_file, "w") as f:
        f.write("""
import pickle
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass

@dataclass
class SoundSource:
    x: int
    y: int
    frequency: float = 440.0
    amplitude: float = 1.0
    color: tuple = (255, 255, 0)

@dataclass
class Microphone:
    x: int
    y: int
    color: tuple = (255, 192, 203)  # Pink color for microphone

# Load the data
with open(r"{data_file}", "rb") as f:
    data = pickle.load(f)

# Extract data
corners = data["corners"]
height_meters = data["height"]
room_width = data["room_width"]
room_height = data["room_height"]
nx = data["nx"]
ny = data["ny"]

# Convert tuples back to objects
sources = [SoundSource(x, y, frequency, amplitude, color) 
          for x, y, frequency, amplitude, color in data["sources"]]
microphones = [Microphone(x, y) for x, y in data["microphones"]]

print("Room data loaded successfully")
print(f"Room height: {{height_meters:.2f}} meters")
print(f"Number of corners: {{len(corners)}}")
print(f"Number of sources: {{len(sources)}}")
print(f"Number of microphones: {{len(microphones)}}")

# Create 3D plot
plt.close('all')  # Close any existing figures
fig = plt.figure('3D Room Layout', figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot room corners and walls (bottom)
room_corners = corners + [corners[0]]  # Add first corner again to close the polygon
x_coords = [x * room_width/nx for x, _ in room_corners]
y_coords = [y * room_height/ny for _, y in room_corners]
z_coords = [0] * len(room_corners)

# Plot bottom edges
ax.plot(x_coords, y_coords, z_coords, 'g-', linewidth=2, label='Bottom Edges')

# Plot top edges
ax.plot(x_coords, y_coords, [height_meters] * len(room_corners), 'g-', linewidth=2, label='Top Edges')

# Plot vertical edges
for i in range(len(corners)):
    x = corners[i][0] * room_width/nx
    y = corners[i][1] * room_height/ny
    ax.plot([x, x], [y, y], [0, height_meters], 'g-', linewidth=2)

# Plot sources with vertical guide lines
for i, source in enumerate(sources):
    x = source.x * room_width/nx
    y = source.y * room_height/ny
    z = height_meters/2  # Place sources at mid-height
    ax.scatter([x], [y], [z], c='yellow', marker='*', s=200, label=f'Source {{i+1}}')
    # Add vertical guide line
    ax.plot([x, x], [y, y], [0, height_meters], 'y--', alpha=0.3)

# Plot microphones with vertical guide lines
for i, mic in enumerate(microphones):
    x = mic.x * room_width/nx
    y = mic.y * room_height/ny
    z = height_meters/2  # Place microphones at mid-height
    ax.scatter([x], [y], [z], c='pink', marker='p', s=100, label=f'Mic {{i+1}}')
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
ax.grid(True)

# Set a comfortable initial view angle
ax.view_init(elev=30, azim=-60)

print("3D visualization ready - you can rotate the view with your mouse")
print("Close the window to return to the main application")

# Enable rotation with the mouse - this will block until window is closed
plt.show()
""".format(data_file=data_file.replace("\\", "\\\\")))
    
    # Run the script in a new process
    subprocess.Popen([sys.executable, script_file])

def draw():
    # Start with a red background
    screen.fill((128, 0, 0))  # Dark red background    # Draw wave visualization with lower opacity
    s = pygame.Surface((screen_width, screen_height))
    s.set_alpha(64)  # Make wave visualization more transparent
    s.fill((128, 0, 0))  # Match the red background
      # Get current frequency for visualization scaling
    current_freq = sources[selected_source_index].frequency if sources and selected_source_index < len(sources) else 500
    
    # For higher frequencies, amplify the wave visualization
    # This addresses the visualization issue for high frequencies without changing grid resolution
    # Higher frequencies (>500Hz) have shorter wavelengths that may be less visible at the current grid resolution
    # Scale factor increases as frequency increases, with 500Hz as the baseline (scale = 1.0)
    # The scaling is capped at 10x to prevent excessive amplification
    vis_scale = min(10, max(1, current_freq / 500))
    
    # Draw walls and wave visualization
    for x in range(nx):
        for y in range(ny):
            if walls[x, y] > 0:
                pygame.draw.rect(screen, WALL_COLOR, 
                               (x * scale_x, y * scale_y, scale_x, scale_y))
            else:
                # Apply frequency-based scaling to wave visualization
                intensity = int(((wave[x, y] * vis_scale) + 1) * 127.5)
                intensity = max(0, min(255, intensity))
                pygame.draw.rect(screen, (intensity, 0, 0), 
                               (x * scale_x, y * scale_y, scale_x, scale_y))

    # Draw room corners and lines
    if len(room.corners) > 0:
        scaled_points = []
        for x, y in room.corners:
            screen_x = int(x * scale_x)
            screen_y = int(y * scale_y)
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
                        (source.x * scale_x, source.y * scale_y, scale_x, scale_y))
        pygame.draw.rect(screen, border_color,
                        (source.x * scale_x, source.y * scale_y, scale_x, scale_y), 1)

    # Draw microphones (after sources, before UI)
    for mic in microphones:
        pygame.draw.rect(screen, mic.color, 
                        (mic.x * scale_x, mic.y * scale_y, scale_x, scale_y))

    # Draw current settings - only if there are sources
    small_font = pygame.font.Font(None, 24)
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
    elif height_adjustment_mode:
        mode_text += "Height Adjustment"
    else:
        mode_text += "None"
      # Calculate visualization scaling factor for display
    current_freq = sources[selected_source_index].frequency if sources and selected_source_index < len(sources) else 500
    vis_scale = min(10, max(1, current_freq / 500))
    scale_text = f"Vis Scale: {vis_scale:.1f}x"
    
    freq_surface = small_font.render(freq_text, True, INTENSITY_LINE_COLOR)
    amp_surface = small_font.render(amp_text, True, INTENSITY_LINE_COLOR)
    mode_surface = small_font.render(mode_text, True, INTENSITY_LINE_COLOR)
    scale_surface = small_font.render(scale_text, True, INTENSITY_LINE_COLOR)
    
    screen.blit(freq_surface, (500, 10))
    screen.blit(amp_surface, (500, 35))
    screen.blit(mode_surface, (650, 10))
    screen.blit(scale_surface, (650, 35))# Draw bottom control panel background
    bottom_panel_rect = pygame.Rect(0, screen_height, screen_width, bottom_panel_height)
    pygame.draw.rect(screen, GRID_COLOR, bottom_panel_rect)
    
    # Draw divider line
    pygame.draw.line(screen, INTENSITY_LINE_COLOR, 
                    (0, screen_height),
                    (screen_width, screen_height), 2)

    # Draw material dropdowns
    wall_dropdown.draw(screen)
    ceiling_dropdown.draw(screen) 
    floor_dropdown.draw(screen)    # Create sections in bottom panel
    panel_y = screen_height + 45  # Starting Y position for panel content (slightly higher)
    left_margin = 20
    
    # Material controls help text with more space
    material_help = ""
    material_help_surface = small_font.render(material_help, True, INTENSITY_LINE_COLOR)
    screen.blit(material_help_surface, (left_margin, screen_height + 5))  # Place right below the divider
    
    # Left section: Room Controls
    height_text = f"Room Height: {height_input_text} ft" if height_input_active else f"Room Height: {room_height:.1f} ft ({room_height * FEET_TO_METERS:.1f} m)"
    height_surface = small_font.render(height_text, True, INTENSITY_LINE_COLOR)
    screen.blit(height_surface, (left_margin, panel_y + 15))  # Add padding after material help
      # Room controls help text with adjusted spacing
    room_controls = ["H: Height Mode", "M: Room Mode", "X: Mic Mode"]
    for i, control in enumerate(room_controls):
        room_surface = small_font.render(control, True, INTENSITY_LINE_COLOR)
        screen.blit(room_surface, (left_margin, panel_y + 35 + i*18))  # Reduced vertical spacing
    
    # Middle section: Source Controls with adjusted position
    source_text = "Source Controls:"
    source_controls = ["F: Single Source", "SPACE: All Sources", "S: Add Source"]
    source_surface = small_font.render(source_text, True, INTENSITY_LINE_COLOR)    
    screen.blit(source_surface, (screen_width//3, panel_y + 15))  # Added padding
    for i, control in enumerate(source_controls):
        control_surface = small_font.render(control, True, INTENSITY_LINE_COLOR)
        screen.blit(control_surface, (screen_width//3, panel_y + 35 + i*18))  # Reduced vertical spacing
      # Right section: Analysis Controls with adjusted position
    analysis_text = "Analysis:"
    analysis_controls = ["C: Calculate", "2: 2D View", "3: 3D View"]
    analysis_surface = small_font.render(analysis_text, True, INTENSITY_LINE_COLOR)    
    screen.blit(analysis_surface, (2*screen_width//3, panel_y + 15))  # Added padding
    for i, control in enumerate(analysis_controls):
        control_surface = small_font.render(control, True, INTENSITY_LINE_COLOR)
        screen.blit(control_surface, (2*screen_width//3, panel_y + 35 + i*18))  # Reduced vertical spacing
      # Additional status info in the bottom row with more space
    status_y = panel_y + 90  # Increased spacing before status text
    if height_adjustment_mode:
        status_text = "HEIGHT ADJUSTMENT MODE - Use Up/Down arrows"
    elif room_drawing_mode:
        status_text = "ROOM DRAWING MODE - Click to place corners"
    elif mic_mode:
        status_text = "MICROPHONE PLACEMENT MODE - Click to place mics"
    else:
        # Show material options for the active dropdown
        active_dropdown = dropdowns[active_dropdown_index]
        status_text = active_dropdown.get_available_options_text()
        
        # Add frequency info if sources are active
        if any(source.active for source in sources):
            active_source = sources[selected_source_index]
            status_text += f" | Frequency: {active_source.frequency} Hz"
    
    status_surface = small_font.render(status_text, True, INTENSITY_LINE_COLOR)    
    status_rect = status_surface.get_rect(center=(screen_width//2, status_y + 10))
    screen.blit(status_surface, status_rect)

    pygame.display.flip()

def erase_at_position(x: int, y: int):
    global sources, selected_source_index
    # First try to erase any sources at this position
    for i, source in enumerate(reversed(sources)):
        if source.x == x and source.y == y:
            if len(sources) > 1:  # Keep at least one source
                sources.pop(len(sources) - 1 - i)
                selected_source_index = min(selected_source_index, len(sources) - 1)
                return True    # If no source was erased, erase walls
    if 0 <= x < nx and 0 <= y < ny:
        walls[x, y] = WALL_NONE
    return False

def reset_simulation():
    global wave, wave_prev, walls, sources, selected_source_index
    # Reset wave fields
    wave.fill(0)
    wave_prev.fill(0)    # Reset walls
    walls.fill(WALL_NONE)# Reset sources to single initial source with default frequency
    sources = [SoundSource(nx // 6, ny // 6, 
                           frequency=AVAILABLE_FREQUENCIES[DEFAULT_FREQUENCY_INDEX], 
                           amplitude=1.0, 
                           color=SOURCE_COLORS[0])]
    selected_source_index = 0

def place_sound_source():
    global sources, selected_source_index
    mouse_x, mouse_y = pygame.mouse.get_pos()
    if mouse_y >= screen_height:  # Don't place sources in the bottom panel
        return
    grid_x = int(mouse_x // scale_x)
    grid_y = int(mouse_y // scale_y)
    if 0 <= grid_x < nx and 0 <= grid_y < ny and walls[grid_x, grid_y] == WALL_NONE:
        # Create new source with cycling colors
        new_source = SoundSource(
            x=grid_x, 
            y=grid_y,
            frequency=AVAILABLE_FREQUENCIES[DEFAULT_FREQUENCY_INDEX],
            amplitude=1.0,
            color=SOURCE_COLORS[len(sources) % len(SOURCE_COLORS)]
        )
        sources.append(new_source)
        selected_source_index = len(sources) - 1

# Main loop
running = True
height_adjustment_mode = False  # Track if we're adjusting height

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_f:
                # Toggle source activation on/off instead of just activating
                sources[selected_source_index].active = not sources[selected_source_index].active
                print(f"Source {selected_source_index+1} {'activated' if sources[selected_source_index].active else 'deactivated'}")
            elif event.key == pygame.K_SPACE:
                # Toggle all sources on/off
                all_active = all(source.active for source in sources)
                for source in sources:
                    source.active = not all_active
                print(f"All sources {'deactivated' if all_active else 'activated'}")
            elif event.key == pygame.K_h:
                height_adjustment_mode = True            
            elif event.key == pygame.K_UP:
                if height_adjustment_mode:
                    # Convert step from meters to feet
                    room_height = min(50.0, room_height + HEIGHT_STEP / FEET_TO_METERS)
                    print(f"Room height: {room_height * FEET_TO_METERS:.1f} meters")
                else:
                    # Find current frequency index and go to next frequency
                    current_freq = sources[selected_source_index].frequency
                    # Find closest idx in our list
                    closest_idx = min(range(len(AVAILABLE_FREQUENCIES)), 
                                    key=lambda i: abs(AVAILABLE_FREQUENCIES[i] - current_freq))
                    # Move to next frequency (or stay at max)
                    next_idx = min(closest_idx + 1, len(AVAILABLE_FREQUENCIES) - 1)
                    sources[selected_source_index].frequency = AVAILABLE_FREQUENCIES[next_idx]
                    print(f"Frequency: {sources[selected_source_index].frequency} Hz")
            elif event.key == pygame.K_DOWN:
                if height_adjustment_mode:
                    # Convert step from meters to feet
                    room_height = max(2.0, room_height - HEIGHT_STEP / FEET_TO_METERS)
                    print(f"Room height: {room_height * FEET_TO_METERS:.1f} meters")
                else:
                    # Find current frequency index and go to previous frequency
                    current_freq = sources[selected_source_index].frequency
                    # Find closest frequency index in our list
                    closest_idx = min(range(len(AVAILABLE_FREQUENCIES)), 
                                    key=lambda i: abs(AVAILABLE_FREQUENCIES[i] - current_freq))
                    # Move to previous frequency (or stay at min)
                    prev_idx = max(closest_idx - 1, 0)
                    sources[selected_source_index].frequency = AVAILABLE_FREQUENCIES[prev_idx]
                    print(f"Frequency: {sources[selected_source_index].frequency} Hz")
            elif event.key == pygame.K_RIGHT:
                sources[selected_source_index].amplitude = min(5.0, sources[selected_source_index].amplitude + AMP_STEP)
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
            elif event.key == pygame.K_2:
                visualize_room_layout()            
            elif event.key == pygame.K_3:  # Press '3' for 3D view
                visualize_room_3d()
            # Dropdown navigation with Tab and material selection with number keys
            elif event.key == pygame.K_TAB and pygame.key.get_mods() & pygame.KMOD_SHIFT:
                # Shift+Tab to cycle backwards through dropdowns
                for dropdown in dropdowns:
                    dropdown.set_active(False)
                active_dropdown_index = (active_dropdown_index - 1) % len(dropdowns)
                dropdowns[active_dropdown_index].set_active(True)
            elif event.key == pygame.K_TAB:
                # Tab to cycle through dropdowns
                for dropdown in dropdowns:
                    dropdown.set_active(False)
                active_dropdown_index = (active_dropdown_index + 1) % len(dropdowns)
                dropdowns[active_dropdown_index].set_active(True)            # Number keys 1-8 for selecting material options when a dropdown is active
            elif pygame.K_1 <= event.key <= pygame.K_8:
                option_index = event.key - pygame.K_1  # Convert key to 0-7 index
                current_dropdown = dropdowns[active_dropdown_index]
                # Only change if the option index is valid for this dropdown
                if option_index < len(current_dropdown.options):
                    current_dropdown.selected_index = option_index
                    material_name = current_dropdown.options[option_index][0]
                    print(f"Selected {current_dropdown.label}: {material_name}")
            
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
            # Remove the source deactivation from key release
            # so sources stay active until toggled off
            if event.key == pygame.K_h:
                height_adjustment_mode = False
        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                # Check if any dropdown was clicked
                dropdown_clicked = False
                for i, dropdown in enumerate(dropdowns):
                    if dropdown.handle_event(event):
                        # Set this dropdown as active
                        for j, d in enumerate(dropdowns):
                            d.set_active(j == i)
                        active_dropdown_index = i
                        dropdown_clicked = True
                        break
                if not dropdown_clicked:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    if mouse_y < screen_height:  # Only handle clicks in the main area
                        grid_x = int(mouse_x // scale_x)
                        grid_y = int(mouse_y // scale_y)
                        if 0 <= grid_x < nx and 0 <= grid_y < ny:
                            if room_drawing_mode:
                                room.add_corner(grid_x, grid_y)
                                room.is_drawing = True
                            elif mic_mode:
                                microphones.append(Microphone(grid_x, grid_y))
                                print(f"Microphone placed at grid position ({grid_x}, {grid_y})")

    update_wave()
    draw()
    clock.tick(144)

pygame.quit()
sys.exit()
