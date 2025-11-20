import bpy
import os

##################################################################
#  
#   Bibliografy:
#    Claude: https://claude.ai/chat/aa6284f1-c7fb-4f61-8949-883d4f91b842
#
#
#
##################################################################

# Configuration - Update these paths to your files
OBJ_FILE_1 = "path/to/your/first_model.obj"
OBJ_FILE_2 = "path/to/your/second_model.obj"
TEXTURE_FILE = "path/to/your/texture.png"
OUTPUT_FBX = "path/to/output/model_with_texture.fbx"

def clear_scene():
    """Remove all objects from the scene"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # Clear orphaned data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)

def import_obj(filepath):
    """Import OBJ file"""
    bpy.ops.wm.obj_import(filepath=filepath)
    return bpy.context.selected_objects

def create_material_with_texture(texture_path, material_name="TexturedMaterial"):
    """Create a material with texture that works in both Blender and Unreal"""
    
    # Create new material
    mat = bpy.data.materials.new(name=material_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Clear default nodes
    nodes.clear()
    
    # Create nodes
    node_output = nodes.new(type='ShaderNodeOutputMaterial')
    node_output.location = (400, 0)
    
    node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    node_bsdf.location = (0, 0)
    
    node_tex = nodes.new(type='ShaderNodeTexImage')
    node_tex.location = (-400, 0)
    
    # Load texture image
    if os.path.exists(texture_path):
        img = bpy.data.images.load(texture_path)
        node_tex.image = img
    else:
        print(f"Warning: Texture file not found: {texture_path}")
        return mat
    
    # Connect nodes
    links.new(node_tex.outputs['Color'], node_bsdf.inputs['Base Color'])
    links.new(node_bsdf.outputs['BSDF'], node_output.inputs['Surface'])
    
    return mat

def apply_material_to_objects(objects, material):
    """Apply material to all objects"""
    for obj in objects:
        if obj.type == 'MESH':
            # Clear existing materials
            obj.data.materials.clear()
            # Add new material
            obj.data.materials.append(material)
            print(f"Material applied to {obj.name}")

def export_fbx(filepath):
    """Export scene to FBX with proper settings"""
    bpy.ops.export_scene.fbx(
        filepath=filepath,
        use_selection=False,
        global_scale=1.0,
        apply_unit_scale=True,
        apply_scale_options='FBX_SCALE_NONE',
        bake_space_transform=False,
        object_types={'MESH'},
        use_mesh_modifiers=True,
        mesh_smooth_type='FACE',
        use_mesh_edges=False,
        use_tspace=True,
        use_custom_props=False,
        add_leaf_bones=False,
        primary_bone_axis='Y',
        secondary_bone_axis='X',
        use_armature_deform_only=False,
        armature_nodetype='NULL',
        bake_anim=False,
        path_mode='COPY',  # This embeds textures
        embed_textures=True,  # Embed textures in FBX
        batch_mode='OFF',
        use_batch_own_dir=True,
        axis_forward='-Z',
        axis_up='Y'
    )
    print(f"FBX exported to: {filepath}")

def main():
    """Main execution function"""
    
    print("Starting OBJ to FBX conversion with texture...")
    
    # Validate file paths
    if not os.path.exists(OBJ_FILE_1):
        print(f"Error: OBJ file 1 not found: {OBJ_FILE_1}")
        return
    if not os.path.exists(OBJ_FILE_2):
        print(f"Error: OBJ file 2 not found: {OBJ_FILE_2}")
        return
    if not os.path.exists(TEXTURE_FILE):
        print(f"Error: Texture file not found: {TEXTURE_FILE}")
        return
    
    # Clear the scene
    clear_scene()
    
    # Import OBJ files
    print(f"Importing {OBJ_FILE_1}...")
    objects_1 = import_obj(OBJ_FILE_1)
    
    print(f"Importing {OBJ_FILE_2}...")
    objects_2 = import_obj(OBJ_FILE_2)
    
    all_objects = objects_1 + objects_2
    
    # Create material with texture
    print("Creating material with texture...")
    material = create_material_with_texture(TEXTURE_FILE)
    
    # Apply material to all imported objects
    print("Applying material to objects...")
    apply_material_to_objects(all_objects, material)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(OUTPUT_FBX)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Export to FBX
    print("Exporting to FBX...")
    export_fbx(OUTPUT_FBX)
    
    print("Conversion complete!")
    print(f"Output file: {OUTPUT_FBX}")

# Run the script
if __name__ == "__main__":
    main()