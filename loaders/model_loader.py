"""Unified model loader with auto-format detection."""
import os
from .obj_loader import load_obj
from .gltf_loader import load_gltf, load_glb


def load_model(filename):
    """Load 3D model from file with automatic format detection.
    
    Supports: .obj, .gltf, .glb
    
    Args:
        filename: Path to the model file
    
    Returns:
        Tuple of (vertices, faces, materials, uvs, textures)
        - vertices: numpy array of shape (N, 3) with vertex positions
        - faces: list of (triangle_indices, material_name) tuples
        - materials: dict mapping material names to {'color': [RGB], 'texture_index': int or None}
        - uvs: numpy array of shape (N, 2) with texture coordinates, or None
        - textures: dict mapping texture indices to PIL.Image objects, or None
    
    Raises:
        ValueError: If file format is not supported
        FileNotFoundError: If file doesn't exist
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Model file not found: {filename}")
    
    # Get file extension
    _, ext = os.path.splitext(filename.lower())
    
    # Route to appropriate loader
    if ext == '.obj':
        print(f"Loading OBJ model: {filename}")
        return load_obj(filename)
    elif ext == '.gltf':
        print(f"Loading GLTF model: {filename}")
        return load_gltf(filename)
    elif ext == '.glb':
        print(f"Loading GLB model: {filename}")
        return load_glb(filename)
    else:
        raise ValueError(
            f"Unsupported model format: {ext}. "
            f"Supported formats: .obj, .gltf, .glb"
        )
