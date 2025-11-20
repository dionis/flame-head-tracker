import argparse
import json
import os
import time
import sys

module_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(module_dir)

from generate_arkit_flame_meshes import Tracker3DImage


print("Tracker3DImage imported successfully!")

### 2d files and 3D object to transforms

converter = Tracker3DImage()

def main():
    parser = argparse.ArgumentParser(description="Create avatar from input.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input image or video.")
    parser.add_argument("--input_type", type=str, required=True, choices=["image", "video", "webcam_frame"], help="Type of input (image, video, webcam_frame).")
    parser.add_argument("--output_dir", type=str, default="generated_avatars", help="Directory to save generated avatars.")
    args = parser.parse_args()

    output_file_path = os.path.join(args.output_dir, f"avatar_{os.path.basename(args.input_path)}.json")
    
    # Simulate avatar creation and return some dummy data
    response = {
        "status": "success",
        "message": f"Avatar created from {args.input_type} at {args.input_path}",
        "output_path": output_file_path,
        "input_type": args.input_type
    }
    print(json.dumps(response))
    
    print("<=== Execute task files ====>")
    
    #result = subprocess.run(command, capture_output=True, text=True, check=True)
    start_time = time.time() 

    result = converter.process_image_to_3d_object(
            input_path = args.input_path,
            input_type = args.input_type,
            output_dir = args.output_dir
        ) 
      
    end_time = time.time() # record the

if __name__ == "__main__":
    main()
