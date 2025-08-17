## Detección de Rostros con MediaPipe + Gradio (Hugging Face Space)

Proyecto para crear una Space en Hugging Face con detección de rostros usando MediaPipe y una interfaz en Gradio. Permite:

- Subir imágenes
- Subir videos (genera un video anotado)
- Usar la webcam del navegador (stream en vivo)

### Estructura
```
webcam_facedetection/
├─ app.py
├─ requirements.txt
├─ utils/
│  └─ face_analyzer.py
└─ README.md
```

### Ejecutar localmente
1) Instalar dependencias:
```bash
pip install -r requirements.txt
```
2) Ejecutar:
```bash
python app.py
```

### Despliegue en Hugging Face Spaces
1) Crear una Space (SDK: Gradio).
2) Subir los archivos del proyecto.
3) El Space se ejecutará automáticamente y mostrará la interfaz.

### Notas
- El video de salida se guarda junto al archivo original con sufijo `.annotated.mp4`.
- La opción de refinar landmarks (iris) incrementa el costo computacional.

### Licencia
MIT


