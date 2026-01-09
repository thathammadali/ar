"""AR Application - Modular version with multi-format model support.

Supports .obj, .gltf, and .glb 3D model formats with material loading.
"""
import cv2
import numpy as np
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, QUIT
from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_LEQUAL,
    GL_LINEAR,
    GL_MODELVIEW,
    GL_PROJECTION,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    glBindTexture,
    glClear,
    glClearDepth,
    glDepthFunc,
    glEnable,
    glGenTextures,
    glLoadIdentity,
    glMatrixMode,
    glMultMatrixf,
    glPopMatrix,
    glPushMatrix,
    glScalef,
    glTexParameteri,
    glTranslatef,
)
from OpenGL.GL import glRotatef
from OpenGL.GLU import gluPerspective

# Import configuration
from config import (
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    FRAME_SKIP,
    TEST_MODE,
    POSITION_SMOOTHING,
    BASE_MODEL_SIZE,
    MARKER_MAPPING
)

# Import modules
from loaders.model_loader import load_model
from detection.marker_detector import MarkerDetector
from rendering.background import render_background
from rendering.model_renderer import draw_model, cv_to_gl


def main():
    """Main application entry point."""
    print("="* 60)
    print("AR Application - Multi-Marker Support")
    print("Supported formats: .obj, .gltf, .glb")
    print("=" * 60)
    
    # Load 3D models from MARKER_MAPPING
    model_library = {}
    unique_models = set(MARKER_MAPPING.values())
    
    print(f"\nPre-loading {len(unique_models)} unique items from mapping...")
    
    for model_file in unique_models:
        print(f"\nLoading: {model_file}")
        try:
            vertices, faces, materials, uvs, textures = load_model(model_file)
            
            # Auto-center vertices
            if len(vertices) > 0:
                center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
                vertices -= center
                print(f"  ✓ Centered model by: {center}")
                
            
            # Store model data + empty texture cache
            # The texture cache maps local texture indices to OpenGL IDs unique to this model
            model_library[model_file] = (vertices, faces, materials, uvs, textures, {})
            print(f"  ✓ Successfully loaded")
            
        except Exception as e:
            print(f"  ✗ Error loading {model_file}: {e}")
            # We don't exit; getting one model wrong shouldn't kill the app if others work
            
    if not model_library:
        print("✗ No models could be loaded. Exiting.")
        return

    # Initialize marker detector
    print("\nInitializing marker detector...")
    try:
        detector = MarkerDetector()
        print("✓ Marker detector ready")
    except Exception as e:
        print(f"✗ Error initializing detector: {e}")
        return
    
    # Initialize Pygame and OpenGL
    print("\nInitializing OpenGL...")
    pygame.init()
    display = (CAMERA_WIDTH, CAMERA_HEIGHT)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("AR Application - Multi-Marker Support")
    
    # Set up depth testing properly
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glClearDepth(1.0)
    
    # Create texture for camera feed
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glEnable(GL_TEXTURE_2D)
    
    print(f"✓ OpenGL initialized: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    
    # Initialize camera
    print("\nInitializing camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Error: Could not open camera")
        return
    print("✓ Camera ready")
    
    print("\n" + "=" * 60)
    print("Application started! Show any configured marker.")
    if TEST_MODE:
        print("TEST MODE: Displaying rotating cube (no marker needed)")
    print("Press ESC or close window to exit.")
    print("=" * 60 + "\n")
    
    # Main loop
    running = True
    frame_count = 0
    
    # Tracking state
    current_scale = BASE_MODEL_SIZE
    smoothed_rvec = None
    smoothed_tvec = None
    current_marker = None
    missing_frames = 0
    MAX_MISSING_FRAMES = 10  # Keep model visible for N frames if marker lost
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_UP:
                    current_scale += 0.05
                    print(f"Scale increased: {current_scale:.2f}")
                elif event.key == pygame.K_DOWN:
                    current_scale = max(0.01, current_scale - 0.05)
                    print(f"Scale decreased: {current_scale:.2f}")
        
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame to match display size
        frame = cv2.resize(frame, display)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detection logic
        # Returns (homography, marker_name) or (None, None)
        homography, marker_name = detector.detect_marker(gray)
        
        detected = False
        rvec, tvec = None, None
        
        if homography is not None:
            try:
                # Get pose for specific marker (needs specific dimensions)
                rvec, tvec = detector.get_pose(homography, marker_name)
                detected = True
                missing_frames = 0
                
                # If we switched markers, reset smoothing to avoid "jumping"
                if marker_name != current_marker:
                    smoothed_rvec = None
                    smoothed_tvec = None
                    current_marker = marker_name
                    # print(f"Switched to marker: {marker_name}")
                    
            except Exception as e:
                print(f"Pose error: {e}")
        
        # Update tracking state (Smoothing & Persistence)
        if detected:
            if smoothed_rvec is None:
                smoothed_rvec = rvec
                smoothed_tvec = tvec
            else:
                # Apply exponential smoothing
                alpha = POSITION_SMOOTHING
                smoothed_rvec = alpha * rvec + (1 - alpha) * smoothed_rvec
                smoothed_tvec = alpha * tvec + (1 - alpha) * smoothed_tvec
        else:
            missing_frames += 1
            if missing_frames > MAX_MISSING_FRAMES:
                smoothed_rvec = None
                smoothed_tvec = None
                current_marker = None # Lost tracking completely
        
        frame_count += 1
        
        # Clear buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Render camera feed as background
        render_background(frame, texture_id)
        
        # TEST MODE: Draw cube at fixed position for debugging
        if TEST_MODE:
             # (... Test logic ...)
            pass 
        
        # AR MODE: Draw based on smoothed pose
        elif smoothed_tvec is not None and current_marker is not None:
            try:
                # 1. Identify which model to draw
                target_model_file = MARKER_MAPPING.get(current_marker)
                if target_model_file and target_model_file in model_library:
                    # Unpack including texture_cache
                    vertices, faces, materials, uvs, textures, texture_cache = model_library[target_model_file]
                
                    # 2. Setup Camera
                    fy = detector.camera_matrix[1, 1]
                    fovy = 2 * np.degrees(np.arctan(CAMERA_HEIGHT / (2 * fy)))
                    
                    glMatrixMode(GL_PROJECTION)
                    glLoadIdentity()
                    gluPerspective(fovy, display[0] / display[1], 0.1, 100.0)
                    
                    # 3. Setup ModelView
                    glMatrixMode(GL_MODELVIEW)
                    glLoadIdentity()
                    view_matrix = cv_to_gl(smoothed_rvec, smoothed_tvec)
                    glMultMatrixf(view_matrix.T)
                    
                    # 4. Transform Model
                    glRotatef(90, 1, 0, 0) # User preference
                    scale = current_scale
                    glScalef(scale, scale, scale)
                    
                    # 5. Draw
                    draw_model(vertices, faces, materials, uvs, textures, texture_cache)
            
            except Exception as e:
                if frame_count % 30 == 0:
                    print(f"Warning: Rendering error: {e}")
        
        pygame.display.flip()
    
    # Cleanup
    cap.release()
    pygame.quit()
    print("\nApplication closed.")


if __name__ == "__main__":
    main()
