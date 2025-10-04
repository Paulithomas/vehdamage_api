# VehDamage API — Detección/Segmentación de daños vehiculares (YOLO + FastAPI)

API ligera para subir una imagen de un vehículo y obtener **cajas/máscaras** con su **probabilidad**, usando un modelo YOLO entrenado en mi tesis. Incluye una mini UI (`/ui`) para probar rápido desde navegador o teléfono.

Subes una **foto** de un auto y la API devuelve una **imagen anotada** (cajas y/o máscaras) y un **JSON** con clase y probabilidad. Incluye una **mini página web** (`/ui`) para probar sin instalar front-end.

---

## 🚀 Demo local 

### Requisitos
- Python 3.10+ y `pip`

### Pasos
```bash
# 1) Clonar y entrar
git clone https://github.com/Paulithomas/vehdamage_api.git
cd vehdamage_api

# 2) Entorno e instalación
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3) Modelo
# Copia tu modelo entrenado a:
#   weights/best.pt
# (Si el peso es "seg", verás máscaras; si es de detección, solo cajas)

# 4) Levantar la API
uvicorn app:app --reload --host 0.0.0.0 --port 8000
