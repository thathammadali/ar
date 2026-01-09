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
    MODEL_FILE,
    POSITION_SMOOTHING,
    BASE_MODEL_SIZE,
)

# Import modules
from loaders.model_loader import load_model
from detection.marker_detector import MarkerDetector
from rendering.background import render_background
from rendering.model_renderer import draw_model, cv_to_gl


def main():
    """Main application entry point."""
    print("="* 60)
    print("AR Application - Multi-Format Model Support")
    print("Supported formats: .obj, .gltf, .glb")
    print("=" * 60)
    
    # Load 3D model
    print(f"\nLoading 3D model: {MODEL_FILE}")
    try:
        vertices, faces, materials, uvs, textures = load_model(MODEL_FILE)
        print(f"✓ Model loaded: {len(vertices)} vertices, {len(faces)} faces")
        
        if textures:
            print(f"✓ Textures loaded: {len(textures)} texture(s)")
        
        if materials:
            print(f"✓ Materials loaded: {list(materials.keys())}")
            for mat_name, mat_info in materials.items():
                color = mat_info['color']
                tex_idx = mat_info.get('texture_index')
                if tex_idx is not None:
                    print(f"  - {mat_name}: RGB({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f}) + Texture {tex_idx}")
                else:
                    print(f"  - {mat_name}: RGB({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})")
        
        if len(vertices) > 0:
            # Auto-center vertices
            # This fixes issues where models are exported with origin at corner
            center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
            vertices -= center
            print(f"✓ Model centered. Shifted by: {center}")
            
            print(f"✓ Vertex range: "
                  f"X({vertices[:, 0].min():.2f}, {vertices[:, 0].max():.2f}), "
                  f"Y({vertices[:, 1].min():.2f}, {vertices[:, 1].max():.2f}), "
                  f"Z({vertices[:, 2].min():.2f}, {vertices[:, 2].max():.2f})")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
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
    pygame.display.set_caption("AR Application - Multi-Format Support")
    
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
    print("Application started! Show marker to camera.")
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
        # Optimize: detected only every N frames, or every frame?
        # For smoothing to work best, every frame is preferred if performance allows.
        homography = detector.detect_marker(gray)
        
        detected = False
        rvec, tvec = None, None
        
        if homography is not None:
            try:
                rvec, tvec = detector.get_pose(homography)
                detected = True
                missing_frames = 0
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
        
        frame_count += 1
        
        # Clear buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Render camera feed as background
        render_background(frame, texture_id)
        
        # TEST MODE: Draw cube at fixed position for debugging
        if TEST_MODE:
            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            gluPerspective(45, display[0] / display[1], 0.1, 1000.0)
            
            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glLoadIdentity()
            glTranslatef(0, 0, -5)
            glRotatef(frame_count * 0.5, 1, 1, 0)
            
            draw_model(vertices, faces, materials, uvs, textures)
            
            glPopMatrix()
            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
        
        # AR MODE: Draw based on smoothed pose
        elif smoothed_tvec is not None:
            try:
                # Calculate correct FOV from camera intrinsics
                fy = detector.camera_matrix[1, 1]
                fovy = 2 * np.degrees(np.arctan(CAMERA_HEIGHT / (2 * fy)))
                
                # Set up projection using correct FOV
                glMatrixMode(GL_PROJECTION)
                glLoadIdentity()
                gluPerspective(fovy, display[0] / display[1], 0.1, 100.0)
                
                # Set up modelview using the smoothed pose
                glMatrixMode(GL_MODELVIEW)
                glLoadIdentity()
                
                # Apply pose transformation from CV to OpenGL
                view_matrix = cv_to_gl(smoothed_rvec, smoothed_tvec)
                glMultMatrixf(view_matrix.T)
                
                # Orient model: rotate 90° around X axis (USER PREFERENCE)
                glRotatef(90, 1, 0, 0)
                
                # Apply scaling (user controlled)
                scale = current_scale
                glScalef(scale, scale, scale)
                
                # Draw the 3D model
                draw_model(vertices, faces, materials, uvs, textures)
            
            except Exception as e:
                # Catch rendering errors to prevent crash
                if frame_count % 30 == 0:  # Only print occasionally
                    print(f"Warning: Rendering error: {e}")
        
        pygame.display.flip()
    
    # Cleanup
    cap.release()
    pygame.quit()
    print("\nApplication closed.")


if __name__ == "__main__":
    main()
