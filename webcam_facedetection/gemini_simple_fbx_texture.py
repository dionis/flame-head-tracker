import bpy
import os

##################################################################
#  
#   Bibliografy:
#    Gemini: https://gemini.google.com/app/2062d1998decf409
#
#
#
#



# --- Configuration Section ---
# IMPORTANT: Update these paths to match your file locations!
# Use an absolute path or a path relative to the Blender file.

# The directory where your OBJ and PNG files are located and where the FBX will be saved
file_dir = "/path/to/your/files/" # <--- CHANGE THIS

# Input file names
obj_file_name_1 = "model_part_A.obj"
obj_file_name_2 = "model_part_B.obj"
texture_file_name = "texture_file.png"

# Output FBX file name
fbx_file_name = "Exported_Model_With_Texture.fbx"

# Full paths for use in the script
obj_path_1 = os.path.join(file_dir, obj_file_name_1)
obj_path_2 = os.path.join(file_dir, obj_file_name_2)
texture_path = os.path.join(file_dir, texture_file_name)
fbx_output_path = os.path.join(file_dir, fbx_file_name)

# Name for the new material and image/texture
material_name = "Custom_Material"
image_name = "Model_Texture_Image"
# ------------------------------


def setup_and_export_fbx_with_texture():
    # 1. Clear the scene (optional, but recommended for clean exports)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # 2. Import the OBJ files
    print(f"Importing {obj_file_name_1}...")
    bpy.ops.import_scene.obj(filepath=obj_path_1)
    
    print(f"Importing {obj_file_name_2}...")
    bpy.ops.import_scene.obj(filepath=obj_path_2)

    # Select the imported objects to apply the material later
    imported_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    
    if not imported_objects:
        print("ERROR: No mesh objects were imported. Check your OBJ paths.")
        return

    # 3. Load the texture image
    try:
        # Load or get the image data block
        if image_name in bpy.data.images:
            tex_image = bpy.data.images[image_name]
        else:
            print(f"Loading texture from {texture_path}...")
            tex_image = bpy.data.images.load(texture_path, check_existing=True)
            tex_image.name = image_name
            
    except RuntimeError as e:
        print(f"ERROR: Could not load texture image from {texture_path}. {e}")
        return

    # 4. Create the material and setup nodes (Principled BSDF)
    print(f"Creating material '{material_name}'...")
    if material_name in bpy.data.materials:
        mat = bpy.data.materials[material_name]
    else:
        mat = bpy.data.materials.new(name=material_name)
        
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    
    # Clear default nodes (optional)
    nodes.clear()
    
    # Create nodes: Output, Principled BSDF, Image Texture
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    principled_node = nodes.new(type='ShaderNodeBsdfPrincipled')
    texture_node = nodes.new(type='ShaderNodeTexImage')
    
    # Link the nodes: Texture -> Principled Color, Principled -> Output Surface
    links = mat.node_tree.links
    links.new(texture_node.outputs['Color'], principled_node.inputs['Base Color'])
    links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])
    
    # Assign the loaded image to the texture node
    texture_node.image = tex_image

    # 5. Assign the material to all imported objects
    print("Assigning material to objects...")
    for obj in imported_objects:
        # If the object doesn't have a material slot, add one
        if not obj.data.materials:
            obj.data.materials.append(mat)
        # Otherwise, replace the first slot
        else:
            obj.data.materials[0] = mat

    # 6. Select all objects for export
    bpy.ops.object.select_all(action='DESELECT')
    for obj in imported_objects:
        obj.select_set(True)
        
    # Set the active object (often needed for export context)
    if imported_objects:
        bpy.context.view_layer.objects.active = imported_objects[0]

    # 7. Export to FBX
    print(f"Exporting FBX to {fbx_output_path}...")
    bpy.ops.export_scene.fbx(
        filepath=fbx_output_path,
        use_selection=True,  # Export only selected objects
        bake_anim=False,     # No animation baking
        bake_anim_use_all_actions=False,
        mesh_smooth_type='FACE', # Use FACE for better compatibility
        path_mode='COPY',    # Embed textures in the FBX file (recommended for UE/other tools)
        embed_textures=True  # Ensure textures are embedded
    )
    
    print("✅ **FBX export complete!**")
    print(f"File saved to: {fbx_output_path}")

# Run the main function
setup_and_export_fbx_with_texture()