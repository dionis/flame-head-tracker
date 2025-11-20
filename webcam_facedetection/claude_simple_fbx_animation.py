import bpy
import os

##################################################################
#  
#   Bibliografy:
#    Claude: https://claude.ai/chat/10a00d7b-0e27-4c5c-9258-82cfbcf7de72
#
#
#
#

# Configuration
OBJ_FOLDER = "C:/path/to/your/obj/files"  # Change this to your OBJ folder path
FBX_OUTPUT = "C:/path/to/output/animation.fbx"  # Change this to your desired output path
FRAME_RATE = 24  # Frames per second
NUM_FRAMES = 52  # Number of OBJ files

def clear_scene():
    """Remove all objects from the scene"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def import_obj_sequence():
    """Import OBJ files and create shape key animation"""
    
    # Get sorted list of OBJ files
    obj_files = sorted([f for f in os.listdir(OBJ_FOLDER) if f.endswith('.obj')])
    
    if len(obj_files) == 0:
        print(f"No OBJ files found in {OBJ_FOLDER}")
        return None
    
    print(f"Found {len(obj_files)} OBJ files")
    
    # Import first OBJ as base mesh
    first_obj = os.path.join(OBJ_FOLDER, obj_files[0])
    bpy.ops.import_scene.obj(filepath=first_obj)
    
    # Get the imported object (assumes single mesh per OBJ)
    base_obj = bpy.context.selected_objects[0]
    base_mesh = base_obj.data
    
    # Add basis shape key
    base_obj.shape_key_add(name='Basis', from_mix=False)
    
    # Import remaining OBJs as shape keys
    for i, obj_file in enumerate(obj_files[1:], start=1):
        obj_path = os.path.join(OBJ_FOLDER, obj_file)
        
        # Import OBJ
        bpy.ops.import_scene.obj(filepath=obj_path)
        imported_obj = bpy.context.selected_objects[0]
        
        # Add as shape key to base object
        shape_key = base_obj.shape_key_add(name=f'Frame_{i:03d}', from_mix=False)
        
        # Copy vertex positions from imported mesh to shape key
        for v_idx, vert in enumerate(imported_obj.data.vertices):
            shape_key.data[v_idx].co = vert.co
        
        # Delete the imported object
        bpy.data.objects.remove(imported_obj, do_unlink=True)
        
        print(f"Processed frame {i+1}/{len(obj_files)}")
    
    return base_obj

def animate_shape_keys(obj):
    """Create animation using shape keys"""
    
    if not obj.data.shape_keys:
        print("No shape keys found!")
        return
    
    shape_keys = obj.data.shape_keys.key_blocks
    
    # Set frame range
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = NUM_FRAMES
    bpy.context.scene.render.fps = FRAME_RATE
    
    # Animate each shape key
    for i, sk in enumerate(shape_keys[1:], start=1):  # Skip basis
        # Set all keys to 0
        for other_sk in shape_keys[1:]:
            other_sk.value = 0.0
        
        # Keyframe current shape key at its frame
        sk.value = 1.0
        sk.keyframe_insert(data_path="value", frame=i)
        
        # Keyframe 0 before and after
        if i > 1:
            sk.value = 0.0
            sk.keyframe_insert(data_path="value", frame=i-1)
        
        if i < NUM_FRAMES:
            sk.value = 0.0
            sk.keyframe_insert(data_path="value", frame=i+1)
    
    print("Animation created successfully")

def export_fbx(filepath):
    """Export scene to FBX"""
    bpy.ops.export_scene.fbx(
        filepath=filepath,
        use_selection=False,
        bake_anim=True,
        bake_anim_use_all_actions=False,
        bake_anim_use_nla_strips=False,
        bake_anim_step=1.0,
        bake_anim_simplify_factor=0.0,
        path_mode='AUTO',
        embed_textures=False,
        mesh_smooth_type='FACE',
        use_mesh_modifiers=True
    )
    print(f"FBX exported to {filepath}")

def main():
    """Main execution function"""
    print("Starting OBJ to FBX animation conversion...")
    
    # Clear the scene
    clear_scene()
    
    # Import OBJ sequence
    obj = import_obj_sequence()
    
    if obj is None:
        print("Failed to import OBJ files")
        return
    
    # Create animation
    animate_shape_keys(obj)
    
    # Export to FBX
    export_fbx(FBX_OUTPUT)
    
    print("Process completed successfully!")

# Run the script
if __name__ == "__main__":
    main()