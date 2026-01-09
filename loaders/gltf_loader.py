"""GLTF and GLB file loader with texture support."""
import numpy as np
import io
from PIL import Image


def load_gltf(filename):
    """Load vertices, faces, and textures from GLTF/GLB file.

    Args:
        filename: Path to the GLTF or GLB file

    Returns:
        Tuple of (vertices, faces, materials, uvs, textures)
        - vertices: numpy array (N, 3)
        - faces: list of (triangle_indices, material_name) tuples
        - materials: dict {name: {'color': [R,G,B], 'texture_index': int or None}}
        - uvs: numpy array (N, 2) or None
        - textures: dict {index: PIL.Image} or None
    """
    try:
        from pygltflib import GLTF2
    except ImportError:
        raise ImportError(
            "pygltflib is required for GLTF/GLB support. "
            "Install with: pip install pygltflib"
        )
    
    gltf = GLTF2().load(filename)
    
    all_vertices = []
    all_uvs = []
    all_faces = []
    materials_dict = {}
    textures_dict = {}
    
    # Extract textures first
    print(f"DEBUG: Found {len(gltf.textures) if gltf.textures else 0} textures in GLTF")
    
    if gltf.textures and gltf.images:
        for tex_idx, texture in enumerate(gltf.textures):
            if texture.source is not None:
                image_data = gltf.images[texture.source]
                
                # Get image binary data
                if image_data.bufferView is not None:
                    buffer_view = gltf.bufferViews[image_data.bufferView]
                    buffer = gltf.buffers[buffer_view.buffer]
                    data = gltf.get_data_from_buffer_uri(buffer.uri)
                    
                    offset = buffer_view.byteOffset or 0
                    length = buffer_view.byteLength
                    image_bytes = data[offset:offset + length]
                    
                    # Load image with PIL
                    try:
                        img = Image.open(io.BytesIO(image_bytes))
                        img = img.convert('RGB')  # Ensure RGB format
                        textures_dict[tex_idx] = img
                        print(f"DEBUG: Loaded texture {tex_idx}: {img.size[0]}x{img.size[1]}")
                    except Exception as e:
                        print(f"DEBUG: Failed to load texture {tex_idx}: {e}")
                elif image_data.uri:
                    print(f"DEBUG: External URI textures not yet supported: {image_data.uri}")
    
    # Extract materials with texture references
    print(f"DEBUG: Found {len(gltf.materials) if gltf.materials else 0} materials in GLTF")
    
    if gltf.materials:
        for idx, material in enumerate(gltf.materials):
            mat_name = material.name or f"Material_{idx}"
            
            # Default values
            color = [0.8, 0.8, 0.8]
            texture_index = None
            
            if material.pbrMetallicRoughness:
                pbr = material.pbrMetallicRoughness
                
                # Get color
                if pbr.baseColorFactor and len(pbr.baseColorFactor) >= 3:
                    color = list(pbr.baseColorFactor[:3])
                
                # Get texture
                if pbr.baseColorTexture:
                    texture_index = pbr.baseColorTexture.index
                    print(f"DEBUG: Material '{mat_name}' uses texture {texture_index}")
            
            materials_dict[mat_name] = {
                'color': color,
                'texture_index': texture_index
            }
    else:
        materials_dict["Default"] = {'color': [0.8, 0.8, 0.8], 'texture_index': None}
    
    # Process each mesh
    for mesh_idx, mesh in enumerate(gltf.meshes):
        for prim_idx, primitive in enumerate(mesh.primitives):
            # Get material name
            material_name = None
            if primitive.material is not None and gltf.materials:
                material = gltf.materials[primitive.material]
                material_name = material.name or f"Material_{primitive.material}"
            
            # Get vertex positions
            accessor = gltf.accessors[primitive.attributes.POSITION]
            buffer_view = gltf.bufferViews[accessor.bufferView]
            buffer = gltf.buffers[buffer_view.buffer]
            data = gltf.get_data_from_buffer_uri(buffer.uri)
            
            start_offset = buffer_view.byteOffset or 0
            accessor_offset = accessor.byteOffset or 0
            total_offset = start_offset + accessor_offset
            
            vertex_data = np.frombuffer(
                data,
                dtype=np.float32,
                count=accessor.count * 3,
                offset=total_offset
            ).reshape(-1, 3)
            
            vertex_offset = len(all_vertices)
            all_vertices.extend(vertex_data.tolist())
            
            # Get UV coordinates if available
            uv_data = None
            if hasattr(primitive.attributes, 'TEXCOORD_0') and primitive.attributes.TEXCOORD_0 is not None:
                uv_accessor = gltf.accessors[primitive.attributes.TEXCOORD_0]
                uv_buffer_view = gltf.bufferViews[uv_accessor.bufferView]
                
                uv_offset = (uv_buffer_view.byteOffset or 0) + (uv_accessor.byteOffset or 0)
                
                uv_data = np.frombuffer(
                    data,
                    dtype=np.float32,
                    count=uv_accessor.count * 2,
                    offset=uv_offset
                ).reshape(-1, 2)
            
            # If we have UVs, add them
            if uv_data is not None:
                all_uvs.extend(uv_data.tolist())
            else:
                # Add dummy UVs (0, 0) for vertices without texture coordinates
                all_uvs.extend([[0, 0]] * len(vertex_data))
            
            # Get indices
            if primitive.indices is not None:
                indices_accessor = gltf.accessors[primitive.indices]
                indices_buffer_view = gltf.bufferViews[indices_accessor.bufferView]
                
                indices_offset = (indices_buffer_view.byteOffset or 0) + \
                                (indices_accessor.byteOffset or 0)
                
                component_type = indices_accessor.componentType
                if component_type == 5123:  # UNSIGNED_SHORT
                    indices_dtype = np.uint16
                elif component_type == 5125:  # UNSIGNED_INT
                    indices_dtype = np.uint32
                else:
                    indices_dtype = np.uint16
                
                indices = np.frombuffer(
                    data,
                    dtype=indices_dtype,
                    count=indices_accessor.count,
                    offset=indices_offset
                )
                
                # Group into triangles
                for i in range(0, len(indices), 3):
                    if i + 2 < len(indices):
                        triangle = [
                            int(indices[i]) + vertex_offset,
                            int(indices[i + 1]) + vertex_offset,
                            int(indices[i + 2]) + vertex_offset
                        ]
                        all_faces.append((triangle, material_name))
            else:
                # No indices, use sequential vertices
                for i in range(0, len(vertex_data), 3):
                    if i + 2 < len(vertex_data):
                        triangle = [
                            i + vertex_offset,
                            i + 1 + vertex_offset,
                            i + 2 + vertex_offset
                        ]
                        all_faces.append((triangle, material_name))
    
    vertices_array = np.array(all_vertices, dtype=np.float32)
    uvs_array = np.array(all_uvs, dtype=np.float32) if all_uvs else None
    
    print(f"DEBUG: Loaded {len(vertices_array)} vertices, {len(all_faces)} faces, "
          f"{len(textures_dict)} textures")
    
    return vertices_array, all_faces, materials_dict, uvs_array, textures_dict


# Alias for GLB files
load_glb = load_gltf
