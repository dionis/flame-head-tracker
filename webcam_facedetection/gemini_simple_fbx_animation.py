import bpy
import os

##################################################################
#  
#   Bibliografy:
#    Gemini: https://gemini.google.com/app/e258088c619a7b08
#
#
#
#

# --- Configuration ---
# ⚠️ IMPORTANT: Change this to your directory path
obj_dir = "/path/to/your/52/obj/files/"
# ⚠️ IMPORTANT: Change this to your desired output file path
output_filepath = "/path/to/your/output/animated_file.fbx"

# The naming pattern of your OBJ files (e.g., 'frame_001.obj', 'frame_052.obj')
# Ensure the padding matches your file names (e.g., 3 digits for 001-052)
filename_pattern = "frame_{:03d}.obj"
start_frame = 1
end_frame = 52 # Total number of frames (and OBJ files)

# --- Scene Setup and Cleanup ---

def setup_scene():
    """Cleans the scene and sets up animation parameters."""
    # Deselect all
    bpy.ops.object.select_all(action='DESELECT')

    # Delete all existing objects
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.select_by_type(type='LIGHT')
    bpy.ops.object.select_by_type(type='CAMERA')
    bpy.ops.object.delete()

    # Set scene frame range
    bpy.context.scene.frame_start = start_frame
    bpy.context.scene.frame_end = end_frame
    print("Scene setup complete.")

# --- Import and Keyframing Logic ---

def import_and_keyframe():
    """Imports each OBJ file and keyframes its visibility."""
    all_objs = []

    # 1. Import all 52 OBJ files
    for i in range(start_frame, end_frame + 1):
        frame_num = i
        file_name = filename_pattern.format(frame_num)
        filepath = os.path.join(obj_dir, file_name)

        if not os.path.exists(filepath):
            print(f"⚠️ Error: File not found: {filepath}")
            continue

        print(f"Importing frame {frame_num}: {file_name}")

        # Import the OBJ
        # 'use_smooth_groups' and 'use_split_objects' might need adjustment
        bpy.ops.import_scene.obj(filepath=filepath)

        # Get the newly imported object (assuming only one new object is created)
        # OBJ import can sometimes create multiple objects, but often just one main mesh
        new_objects = [obj for obj in bpy.context.selected_objects]
        
        # It's safer to operate on the active object after import if it's guaranteed
        # For simplicity, we'll collect all meshes created by the last import
        
        # Store all created objects
        for obj in new_objects:
            # Rename the object for better organization
            obj.name = f"Frame_{frame_num:03d}"
            all_objs.append(obj)

            # Keyframe Logic
            # Set the current object to be visible ONLY on its target frame (i)
            # ------------------------------------------------------------------
            
            # Start by making the object invisible in the Viewport (Hide) and Render (Hide_render)
            obj.hide_set(True)
            obj.hide_render = True

            # Insert an initial keyframe for hidden state at the start
            obj.keyframe_insert(data_path='hide', frame=start_frame)
            obj.keyframe_insert(data_path='hide_render', frame=start_frame)
            
            # At the target frame (i), make it visible
            obj.hide_set(False)
            obj.hide_render = False

            # Insert a keyframe for the visible state
            obj.keyframe_insert(data_path='hide', frame=i)
            obj.keyframe_insert(data_path='hide_render', frame=i)

            # At the frame *after* the target frame (i+1), make it hidden again
            if i < end_frame:
                obj.hide_set(True)
                obj.hide_render = True
                obj.keyframe_insert(data_path='hide', frame=i + 1)
                obj.keyframe_insert(data_path='hide_render', frame=i + 1)
                
    print(f"Imported and keyframed {len(all_objs)} objects.")

# --- FBX Export Logic ---

def export_fbx():
    """Selects all animated objects and exports them as FBX."""
    # Deselect all
    bpy.ops.object.select_all(action='DESELECT')
    
    # Select all objects that start with 'Frame_'
    for obj in bpy.context.scene.objects:
        if obj.name.startswith("Frame_"):
            obj.select_set(True)
    
    print(f"Exporting FBX to: {output_filepath}")

    # Export selected objects as FBX
    # Key settings for an animated export:
    # 1. 'use_selection=True' to only export our frame objects
    # 2. 'bake_anim=True' to bake the visibility keyframes into the FBX animation
    # 3. 'bake_anim_use_nla_strips', 'bake_anim_use_all_actions', etc., are good for completeness
    bpy.ops.export_scene.fbx(
        filepath=output_filepath,
        use_selection=True, # Only exports selected objects (our frame meshes)
        add_mesh_smooth_type='FACE', # Can be adjusted
        bake_anim=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend=True, # Use scene start/end frames
        bake_anim_frame_start=start_frame,
        bake_anim_frame_end=end_frame
    )
    print("✨ FBX export complete!")

# --- Main Execution ---

if __name__ == "__main__":
    try:
        setup_scene()
        import_and_keyframe()
        export_fbx()
        print("\n✅ Script execution finished successfully!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")