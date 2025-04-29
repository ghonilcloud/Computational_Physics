import numpy as np
import pygame
import sys

# Keybinds:
# F - Activate sound source
# D - Toggle drawing mode for walls (default: reflective walls)
# E - Toggle erasing mode for walls
# S - Place the sound source at the current mouse position
# R - Switch to drawing reflective walls
# A - Switch to drawing absorptive walls
# Mouse Left Click - Draw or erase walls based on the current mode
# Mouse Left Click (Top Bar) - Select brush size

# Room dimensions (meters)
ROOM_WIDTH = 10.0
ROOM_HEIGHT = 10.0

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
REFLECTIVE = 1
ABSORPTIVE = 2

# Modify walls array to store wall types
walls = np.zeros((nx, ny), dtype=int)  # 0 = no wall, 1 = reflective, 2 = absorptive

# Pygame initialization
pygame.init()
screen_size = 800
bar_height = 60  # Increased height for brush selection
screen = pygame.display.set_mode((screen_size, screen_size + bar_height))
pygame.display.set_caption('Sound Wave Propagation')
clock = pygame.time.Clock()

# Colors
WALL_COLOR = (0, 0, 255)
BACKGROUND_COLOR = (255, 255, 255)
SOUND_COLOR = (255, 0, 0)
SOURCE_COLOR = (255, 255, 0)
BAR_COLOR = (200, 200, 200)
TEXT_COLOR = (0, 0, 0)
BRUSH_COLOR = (100, 100, 100)

# Scaling factor
scale_x = screen_size / nx
scale_y = screen_size / ny

# Sound activation
sound_active = False
decay_rate = 0.99

# Modes
drawing_walls = False
erasing_walls = False
mouse_held = False
brush_size = 1
wall_type = REFLECTIVE  # Default wall type

# Font for status bar
font = pygame.font.Font(None, 36)

# Brush size buttons
brush_sizes = [1, 3, 5]
selected_brush_index = 0

# Update wave function to handle wall types
def update_wave():
    global wave, wave_prev
    wave_next = np.copy(wave)

    if sound_active:
        wave[source_x, source_y] += np.sin(2 * np.pi * pygame.time.get_ticks() / 500.0)

    for x in range(1, nx - 1):
        for y in range(1, ny - 1):
            if walls[x, y] == 0:  # No wall
                wave_next[x, y] = (2 * wave[x, y] - wave_prev[x, y] +
                                   (c * dt / dx) ** 2 *
                                   (wave[x + 1, y] + wave[x - 1, y] +
                                    wave[x, y + 1] + wave[x, y - 1] - 
                                    4 * wave[x, y]))
            elif walls[x, y] == ABSORPTIVE:  # Absorptive wall
                wave_next[x, y] = 0  # Absorb all wave energy

    wave_next *= decay_rate
    wave_prev[:], wave[:] = wave[:], wave_next[:]

    # Reflective walls: Set wave to 0 at reflective wall positions
    wave[walls == REFLECTIVE] = 0

def draw():
    screen.fill(BACKGROUND_COLOR)

    # Draw walls
    for x in range(nx):
        for y in range(ny):
            if walls[x, y] == REFLECTIVE:
                pygame.draw.rect(screen, WALL_COLOR, (x * scale_x, y * scale_y + bar_height, scale_x, scale_y))
            elif walls[x, y] == ABSORPTIVE:
                pygame.draw.rect(screen, (0, 255, 0), (x * scale_x, y * scale_y + bar_height, scale_x, scale_y))  # Green for absorptive walls

    # Draw sound wave
    for x in range(nx):
        for y in range(ny):
            if walls[x, y] == 0:
                intensity = int((wave[x, y] + 1) * 127.5)
                intensity = max(0, min(255, intensity))
                pygame.draw.rect(screen, (intensity, 0, 0), (x * scale_x, y * scale_y + bar_height, scale_x, scale_y))

    # Draw the sound source
    pygame.draw.rect(screen, SOURCE_COLOR, (source_x * scale_x, source_y * scale_y + bar_height, scale_x, scale_y))

    # Draw the top bar
    pygame.draw.rect(screen, BAR_COLOR, (0, 0, screen_size, bar_height))

    # Draw brush selection
    for i, size in enumerate(brush_sizes):
        color = BRUSH_COLOR if i != selected_brush_index else SOUND_COLOR
        pygame.draw.rect(screen, color, (10 + i * 50, 10, 40, 40))
        text = font.render(str(size), True, TEXT_COLOR)
        screen.blit(text, (20 + i * 50, 15))

    pygame.display.flip()

# Modify wall placement to allow different wall types
def modify_walls_at_mouse(wall_type: int):
    mouse_x, mouse_y = pygame.mouse.get_pos()
    if mouse_y < bar_height:
        return  # Ignore clicks on the top bar
    grid_x = int(mouse_x // scale_x)
    grid_y = int((mouse_y - bar_height) // scale_y)

    if 0 <= grid_x < nx and 0 <= grid_y < ny:
        half_size = brush_size // 2
        for i in range(-half_size, half_size + 1):
            for j in range(-half_size, half_size + 1):
                if 0 <= grid_x + i < nx and 0 <= grid_y + j < ny:
                    walls[grid_x + i, grid_y + j] = wall_type
                    print(f"Set wall at ({grid_x + i}, {grid_y + j}) to {wall_type}")  # Debugging

def place_sound_source():
    global source_x, source_y
    mouse_x, mouse_y = pygame.mouse.get_pos()
    if mouse_y < bar_height:
        return
    grid_x = int(mouse_x // scale_x)
    grid_y = int((mouse_y - bar_height) // scale_y)

    if 0 <= grid_x < nx and 0 <= grid_y < ny and walls[grid_x, grid_y] == 0:
        source_x, source_y = grid_x, grid_y

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_f:
                sound_active = True
            elif event.key == pygame.K_d:
                drawing_walls = not drawing_walls
                erasing_walls = False
                wall_type = REFLECTIVE  # Default to reflective walls
            elif event.key == pygame.K_e:
                erasing_walls = not erasing_walls
                drawing_walls = False
            elif event.key == pygame.K_s:
                place_sound_source()
            elif event.key == pygame.K_r:  # Press 'R' for reflective walls
                drawing_walls = True
                erasing_walls = False
                wall_type = REFLECTIVE
            elif event.key == pygame.K_a:  # Press 'A' for absorptive walls
                drawing_walls = True
                erasing_walls = False
                wall_type = ABSORPTIVE
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_f:
                sound_active = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                if mouse_y < bar_height:
                    selected_brush_index = mouse_x // 50
                    selected_brush_index = min(selected_brush_index, len(brush_sizes) - 1)
                    brush_size = brush_sizes[selected_brush_index]
                else:
                    mouse_held = True
                    if drawing_walls:
                        modify_walls_at_mouse(wall_type)
                    elif erasing_walls:
                        modify_walls_at_mouse(0)  # Erase walls
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                mouse_held = False
        elif event.type == pygame.MOUSEMOTION:
            if mouse_held:
                if drawing_walls:
                    modify_walls_at_mouse(wall_type)
                elif erasing_walls:
                    modify_walls_at_mouse(0)  # Erase walls

    update_wave()
    draw()
    clock.tick(60)

pygame.quit()
sys.exit()
