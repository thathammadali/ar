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
    last_homography = None
    
    # Smoothed position for stable tracking
    smoothed_x = 0.0
    smoothed_y = 0.0
    first_detection = True  # Flag to initialize smoothed position on first detection
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame to match display size
        frame = cv2.resize(frame, display)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Only process detection every FRAME_SKIP frames for performance
        if frame_count % FRAME_SKIP == 0:
            last_homography = detector.detect_marker(gray)
            # Reset smoothing on marker loss
            if last_homography is None:
                first_detection = True
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
        
        # AR MODE: Draw based on marker detection
        elif last_homography is not None:
            try:
                rvec, tvec = detector.get_pose(last_homography)
                
                # Get marker center in image coordinates
                marker_corners = np.array([
                    [0, 0], [detector.marker_width, 0],
                    [detector.marker_width, detector.marker_height],
                    [0, detector.marker_height]
                ], dtype=np.float32).reshape(-1, 1, 2)
                img_corners = cv2.perspectiveTransform(marker_corners, last_homography)
                center_x = img_corners[:, 0, 0].mean()
                center_y = img_corners[:, 0, 1].mean()
                
                # Initialize smoothed position on first detection
                if first_detection:
                    smoothed_x = center_x
                    smoothed_y = center_y
                    first_detection = False
                
                # Apply exponential smoothing to reduce jitter
                smoothed_x = POSITION_SMOOTHING * center_x + (1 - POSITION_SMOOTHING) * smoothed_x
                smoothed_y = POSITION_SMOOTHING * center_y + (1 - POSITION_SMOOTHING) * smoothed_y
                
                # Convert to normalized device coordinates
                ndc_x = (smoothed_x / CAMERA_WIDTH) * 2 - 1
                ndc_y = -((smoothed_y / CAMERA_HEIGHT) * 2 - 1)
                
                # Calculate marker size for dynamic scaling
                # Larger marker in image = closer to camera = model should be larger
                marker_width_pixels = np.linalg.norm(img_corners[1, 0] - img_corners[0, 0])
                marker_height_pixels = np.linalg.norm(img_corners[2, 0] - img_corners[1, 0])
                avg_marker_size = (marker_width_pixels + marker_height_pixels) / 2
                
                # Dynamic scale: proportional to marker size
                # Reference: when marker is 200 pixels, use BASE_MODEL_SIZE
                reference_marker_size = 200.0
                dynamic_scale = BASE_MODEL_SIZE * (avg_marker_size / reference_marker_size)
                
                # Set up projection
                glMatrixMode(GL_PROJECTION)
                glPushMatrix()
                glLoadIdentity()
                gluPerspective(45, display[0] / display[1], 0.1, 1000.0)
                
                # Set up modelview
                glMatrixMode(GL_MODELVIEW)
                glPushMatrix()
                glLoadIdentity()
                
                # Position the model at marker location
                glTranslatef(ndc_x * 2, ndc_y * 2, -5)
                
                # Apply rotation from pose estimation
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                angle = np.linalg.norm(rvec)
                if angle > 0:
                    axis = rvec.flatten() / angle
                    glRotatef(np.degrees(angle), axis[0], axis[1], axis[2])
                
                # Orient model: rotate -90° around X axis so -Z sits on marker plane
                glRotatef(-90, 1, 0, 0)
                
                # Flip model to face camera (180° around Y-axis)
                # glRotatef(180, 0, 1, 0)
                
                # Apply dynamic scaling
                glScalef(dynamic_scale, dynamic_scale, dynamic_scale)
                
                # Draw the 3D model
                draw_model(vertices, faces, materials, uvs, textures)
                
                # Restore matrices
                glPopMatrix()
                glMatrixMode(GL_PROJECTION)
                glPopMatrix()
                glMatrixMode(GL_MODELVIEW)
            
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
