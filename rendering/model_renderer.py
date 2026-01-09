"""3D model rendering with materials and texture support."""
import cv2
import numpy as np
from OpenGL.GL import (
    GL_BLEND,
    GL_CLAMP_TO_EDGE,
    GL_CULL_FACE,
    GL_DEPTH_TEST,
    GL_FILL,
    GL_FRONT_AND_BACK,
    GL_LINEAR,
    GL_RGB,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_TRIANGLES,
    GL_UNSIGNED_BYTE,
    glBegin,
    glBindTexture,
    glColor3f,
    glColor4f,
    glDepthMask,
    glDisable,
    glEnable,
    glEnd,
    glGenTextures,
    glPolygonMode,
    glTexCoord2f,
    glTexImage2D,
    glTexParameteri,
    glVertex3fv,
)
from config import MODEL_SCALE


# Global draw counter (legacy cache fallback)
DRAW_COUNT = 0
GLOBAL_TEXTURE_CACHE = {} 


def load_texture_to_opengl(texture_index, image, texture_cache=None):
    """Load a PIL Image as an OpenGL texture.
    
    Args:
        texture_index: Index to cache the texture under
        image: PIL Image object
        texture_cache: Dict to store OpenGL IDs in (optional)
    
    Returns:
        OpenGL texture ID
    """
    if texture_cache is None:
        global GLOBAL_TEXTURE_CACHE
        texture_cache = GLOBAL_TEXTURE_CACHE
    
    if texture_index in texture_cache:
        return texture_cache[texture_index]
    
    # Convert PIL image to bytes
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        img_data = np.array(image, dtype=np.uint8)
    except Exception as e:
        print(f"Error converting texture {texture_index}: {e}")
        return 0

    height, width = img_data.shape[:2]
    
    # Generate OpenGL texture
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    
    # Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    
    # Upload texture data
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGB,
        width, height,
        0, GL_RGB, GL_UNSIGNED_BYTE,
        img_data
    )
    
    # Cache it
    texture_cache[texture_index] = texture_id
    
    print(f"DEBUG: Loaded OpenGL texture {texture_index} -> ID {texture_id}")
    
    return texture_id


def cv_to_gl(rvec, tvec):
    """Convert OpenCV pose to OpenGL view matrix.
    
    Args:
        rvec: Rotation vector from OpenCV
        tvec: Translation vector from OpenCV
    
    Returns:
        4x4 view matrix for OpenGL
    """
    rotation_matrix, _jacobian = cv2.Rodrigues(rvec)
    
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = rotation_matrix
    view[:3, 3] = tvec.flatten()
    
    # Fix coordinate system
    flip = np.diag([1, -1, -1, 1])
    view = flip @ view
    return view


def draw_model(vertices, faces, materials, uvs=None, textures=None, texture_cache=None):
    """Draw the 3D model using OpenGL with materials and textures.
    
    Args:
        vertices: Numpy array of vertex positions (N, 3)
        faces: List of (triangle_indices, material_name) tuples
        materials: Dict mapping material names to {'color': [RGB], 'texture_index': int or None}
        uvs: Numpy array of UV coordinates (N, 2), or None
        textures: Dict mapping texture indices to PIL.Image objects, or None
        texture_cache: Dict for caching OpenGL texture IDs (optional)
    """
    global DRAW_COUNT
    DRAW_COUNT += 1
    
    if texture_cache is None:
        global GLOBAL_TEXTURE_CACHE
        texture_cache = GLOBAL_TEXTURE_CACHE
    
    if DRAW_COUNT % 60 == 1:
        # print(f"Drawing model (frame {DRAW_COUNT})...")
        pass
    
    # Disable blending for solid rendering
    glDisable(GL_BLEND)
    
    # Disable backface culling so all faces are visible
    glDisable(GL_CULL_FACE)
    
    # Set polygon mode to filled
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    
    # Ensure depth test is enabled
    glEnable(GL_DEPTH_TEST)
    glDepthMask(True)
    
    # Load all textures if not already cached
    if textures:
        for tex_idx, img in textures.items():
            if tex_idx not in texture_cache:
                load_texture_to_opengl(tex_idx, img, texture_cache)
    
    # Group faces by material and draw
    current_material = None
    current_texture = None
    
    glBegin(GL_TRIANGLES)
    for face_data in faces:
        triangle_indices, material_name = face_data
        
        # Change material/texture when material changes
        if material_name != current_material:
            glEnd()  # End previous batch
            current_material = material_name
            
            # Get material info
            if material_name and material_name in materials:
                mat_info = materials[material_name]
                color = mat_info['color']
                tex_idx = mat_info.get('texture_index')
                
                # Set color
                glColor3f(color[0], color[1], color[2])
                
                # Handle texture
                if tex_idx is not None and textures and tex_idx in textures:
                    # Enable texturing
                    glEnable(GL_TEXTURE_2D)
                    texture_id = texture_cache.get(tex_idx)
                    if texture_id:
                        glBindTexture(GL_TEXTURE_2D, texture_id)
                        current_texture = tex_idx
                else:
                    # Disable texturing, use solid color
                    glDisable(GL_TEXTURE_2D)
                    current_texture = None
            else:
                # Default material
                glColor3f(1.0, 1.0, 1.0)
                glDisable(GL_TEXTURE_2D)
                current_texture = None
            
            glBegin(GL_TRIANGLES)  # Start new batch
        
        # Draw triangle vertices
        for idx in triangle_indices:
            # Apply UV coordinate if we have textures and UVs
            if current_texture is not None and uvs is not None and idx < len(uvs):
                glTexCoord2f(uvs[idx][0], uvs[idx][1])
            
            # Draw vertex
            glVertex3fv(vertices[idx] * MODEL_SCALE)
    glEnd()
    
    # Reset state
    glColor4f(1.0, 1.0, 1.0, 1.0)
    glEnable(GL_TEXTURE_2D)  # Re-enable for background
