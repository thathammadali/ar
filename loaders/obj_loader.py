"""OBJ and MTL file loader."""
import numpy as np


def load_mtl(filename):
    """Load materials from MTL file.
    
    Args:
        filename: Path to the MTL file
    
    Returns:
        Dictionary mapping material names to color tuples (R, G, B)
    """
    materials = {}
    current_material = None
    
    try:
        with open(filename, "r", encoding="utf-8") as mtl_file:
            for line in mtl_file:
                line = line.strip()
                if line.startswith("newmtl "):
                    current_material = line.split()[1]
                    materials[current_material] = [1.0, 1.0, 1.0]  # Default white
                elif line.startswith("Kd ") and current_material:
                    # Diffuse color
                    rgb = list(map(float, line.split()[1:4]))
                    materials[current_material] = rgb
    except FileNotFoundError:
        print(f"Warning: MTL file {filename} not found, using default colors")
    
    return materials


def load_obj(filename):
    """Load vertices and faces from OBJ file.

    Args:
        filename: Path to the OBJ file

    Returns:
        Tuple of (vertices array, faces list with materials, materials dict)
    """
    vertices = []
    faces = []  # Each element: (triangle_indices, material_name)
    current_material = None
    mtl_file = None

    with open(filename, "r", encoding="utf-8") as obj_file:
        for line in obj_file:
            if line.startswith("mtllib "):
                # Found material library reference
                mtl_file = line.split()[1]
            elif line.startswith("usemtl "):
                # Material assignment
                current_material = line.split()[1]
            elif line.startswith("v "):
                vertices.append(list(map(float, line.split()[1:4])))
            elif line.startswith("f "):
                # Parse face indices (handle both triangles and quads)
                face_data = line.split()[1:]
                face_indices = [int(v.split("/")[0]) - 1 for v in face_data]
                
                # If it's a quad, split into two triangles
                if len(face_indices) == 4:
                    # Triangle 1: vertices 0, 1, 2
                    faces.append((
                        [face_indices[0], face_indices[1], face_indices[2]],
                        current_material
                    ))
                    # Triangle 2: vertices 0, 2, 3
                    faces.append((
                        [face_indices[0], face_indices[2], face_indices[3]],
                        current_material
                    ))
                elif len(face_indices) == 3:
                    # Already a triangle
                    faces.append((face_indices, current_material))

    # Load materials
    materials = {}
    if mtl_file:
        materials = load_mtl(mtl_file)
    
    # Convert materials to new format (for consistency with GLTF loader)
    materials_extended = {}
    for name, color in materials.items():
        materials_extended[name] = {'color': color, 'texture_index': None}

    # OBJ files don't support textures in our simple loader
    # Return None for UVs and textures
    return np.array(vertices, dtype=np.float32), faces, materials_extended, None, None
