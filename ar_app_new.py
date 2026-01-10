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
from detection.one_euro_filter import ARPoseFilter
from rendering.background import render_background
from rendering.model_renderer import draw_model, cv_to_gl, compile_display_list


def main():
    """Main application entry point."""
    print("="* 60)
    print("AR Application - Multi-Marker Support")
    print("Supported formats: .obj, .gltf, .glb")
    print("=" * 60)
    
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
                min_v = vertices.min(axis=0)
                max_v = vertices.max(axis=0)
                center = (max_v + min_v) / 2
                
                # Align BASE (min_y) to 0 for proper seating
                shift = center.copy()
                shift[1] = min_v[1]
                
                vertices -= shift
                print(f"  ✓ Centered model by: {shift} (Base aligned)")
                
            
            # Compile to Display List (GPU)
            texture_cache = {}
            print("  - Compiling geometry to GPU Display List...")
            list_id = compile_display_list(vertices, faces, materials, uvs, textures, texture_cache)
            
            # Store model data + cache + display list ID
            model_library[model_file] = (vertices, faces, materials, uvs, textures, texture_cache, list_id)
            print(f"  ✓ Successfully loaded & compiled (List ID: {list_id})")
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

    # Pose Smoothing (1 Euro Filter)
    # Pose Smoothing (1 Euro Filter)
    # min_cutoff: 0.01 = Extremely stable (Rock solid when still)
    # beta: 0.05 = Heavy smoothing (Glides like syrup / "Sticky")
    pose_filter = ARPoseFilter(min_cutoff=0.01, beta=0.05, d_cutoff=1.0)
    
    current_scale = BASE_MODEL_SIZE
    
    # Persistence state
    last_rvec = None
    last_tvec = None
    active_marker = None
    missing_frames = 0
    MAX_MISSING_FRAMES = 30 # 1 second persistence
    
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
                elif event.key == pygame.K_r:
                    pose_filter.reset()
                    active_marker = None
                    last_rvec, last_tvec = None, None
                    missing_frames = MAX_MISSING_FRAMES + 1
                    print("Reset Tracking")
        
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame
        frame = cv2.resize(frame, display)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # --- DETECTION ---
        # Simple, robust per-frame detection
        homography, marker_name = detector.detect_marker(gray)
        
        detected = False
        
        if homography is not None and marker_name:
            try:
                rvec, tvec = detector.get_pose(homography, marker_name, last_rvec, last_tvec)
                
                # Sanity Check: Reject failed solves or extreme values
                valid_pose = False
                if rvec is not None and tvec is not None:
                    # Check for NaNs/Infs
                    if not (np.isnan(rvec).any() or np.isnan(tvec).any() or 
                            np.isinf(rvec).any() or np.isinf(tvec).any()):
                        
                        # Check for reasonable distance (e.g., within 20 units)
                        dist = np.linalg.norm(tvec)
                        if 0.1 < dist < 20.0:
                             valid_pose = True
                
                if valid_pose:
                     # Filter for smoothness
                     rvec, tvec = pose_filter.filter(rvec, tvec)
                     
                     # Update state
                     last_rvec = rvec
                     last_tvec = tvec
                     active_marker = marker_name
                     missing_frames = 0
                     detected = True
            except Exception as e:
                # print(f"Pose processing error: {e}")
                pass
        
        if not detected:
            missing_frames += 1

        frame_count += 1
        
        # Render
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        render_background(frame, texture_id)
        
        if TEST_MODE:
             pass
        elif missing_frames < MAX_MISSING_FRAMES and active_marker:
            if last_rvec is None or last_tvec is None:
                continue

            # Draw using last known valid pose
            try:
                target_model = MARKER_MAPPING.get(active_marker)
                if target_model in model_library:
                    vertices, faces, materials, uvs, textures, texture_cache, list_id = model_library[target_model]
                    
                    # Camera
                    glMatrixMode(GL_PROJECTION)
                    glLoadIdentity()
                    fy = detector.camera_matrix[1, 1]
                    fovy = 2 * np.degrees(np.arctan(CAMERA_HEIGHT / (2 * fy)))
                    gluPerspective(fovy, display[0] / display[1], 0.1, 100.0)
                    
                    # ModelView
                    glMatrixMode(GL_MODELVIEW)
                    glLoadIdentity()
                    view_matrix = cv_to_gl(last_rvec, last_tvec)
                    glMultMatrixf(view_matrix.T)
                    
                    # Transform
                    glRotatef(90, 1, 0, 0)
                    scale = current_scale
                    glScalef(scale, scale, scale)
                    
                    # Draw
                    draw_model(vertices, faces, materials, uvs, textures, texture_cache, display_list_id=list_id)
                    
            except Exception as e:
                if frame_count % 60 == 0:
                    print(f"Render error: {e}")
                    
        pygame.display.flip()
    
    cap.release()
    pygame.quit()
    print("\nApplication closed.")


if __name__ == "__main__":
    main()
