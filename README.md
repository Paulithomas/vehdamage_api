# VehDamage API — Detección/Segmentación de daños vehiculares (YOLO + FastAPI)

API ligera para subir una imagen de un vehículo y obtener **cajas/máscaras** con su **probabilidad**, usando un modelo YOLO entrenado en mi tesis. Incluye una mini UI (`/ui`) para probar rápido desde navegador o teléfono.

## 🚀 Demo rápida
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
