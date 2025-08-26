import argparse
import json
import os

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

if __name__ == "__main__":
    main()
