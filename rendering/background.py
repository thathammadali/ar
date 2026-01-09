"""Background camera feed rendering."""
import cv2
from OpenGL.GL import (
    GL_DEPTH_TEST,
    GL_MODELVIEW,
    GL_PROJECTION,
    GL_QUADS,
    GL_RGB,
    GL_TEXTURE_2D,
    GL_UNSIGNED_BYTE,
    glBegin,
    glBindTexture,
    glDepthMask,
    glDisable,
    glEnable,
    glEnd,
    glLoadIdentity,
    glMatrixMode,
    glTexCoord2f,
    glTexImage2D,
    glVertex3f,
)


def render_background(frame, texture_id):
    """Render camera frame as background texture.
    
    Args:
        frame: BGR frame from camera
        texture_id: OpenGL texture ID
    """
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.flip(frame_rgb, 0)  # Flip vertically for OpenGL
    
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGB,
        frame_rgb.shape[1], frame_rgb.shape[0],
        0, GL_RGB, GL_UNSIGNED_BYTE, frame_rgb
    )
    
    # Draw textured quad covering the screen
    # Disable depth test AND depth writing for background
    glDisable(GL_DEPTH_TEST)
    glDepthMask(False)  # Don't write to depth buffer
    
    # Set up orthographic projection for background
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0)
    glVertex3f(-1, -1, -0.5)
    glTexCoord2f(1, 0)
    glVertex3f(1, -1, -0.5)
    glTexCoord2f(1, 1)
    glVertex3f(1, 1, -0.5)
    glTexCoord2f(0, 1)
    glVertex3f(-1, 1, -0.5)
    glEnd()
    
    # Re-enable depth test and depth writing for 3D objects
    glEnable(GL_DEPTH_TEST)
    glDepthMask(True)
