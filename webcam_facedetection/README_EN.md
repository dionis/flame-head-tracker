## Face Detection with MediaPipe + Gradio (Hugging Face Space)

This project provides a Hugging Face Space using Google MediaPipe for face detection and Gradio for the UI. It supports:

- Image upload
- Video upload (produces an annotated output video)
- Live webcam streaming in the browser
- 3D model visualization (supports OBJ, FBX, GLTF/GLB, STL with textures)

### Features
- Face detection with bounding boxes
- Face Mesh overlay (contours and tessellation)
- Optional iris refinement for more detailed landmarks
- Per-face metrics: bounding box, score, and approximate interocular distance (pixels)

### Project Structure
```
webcam_facedetection/
├─ app.py
├─ requirements.txt
├─ runtime.txt
├─ utils/
│  ├─ __init__.py
│  └─ face_analyzer.py
├─ README.md        # Spanish
└─ README_EN.md     # English
```

### Run Locally
1) Install dependencies:
```bash
pip install -r requirements.txt
```
2) Launch the app:
```bash
python app.py
```
3) Open the local Gradio URL printed in the console.

### Deploy on Hugging Face Spaces
1) Create a new Space (SDK: Gradio).
2) Upload all project files.
3) The Space should auto-start and display the UI.

### Notes
- The annotated output video is saved next to the original with the `.annotated.mp4` suffix.
- Enabling iris refinement increases compute cost.
- Webcam streaming processes frames client-side in the browser and returns metrics per frame.

### License
MIT

### Check other projects sucha as:

- [Facial Emotion Detection](https://github.com/Life-Experimentalist/Facial-Emotion-Detector)

