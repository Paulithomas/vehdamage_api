from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, Response, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
from uuid import uuid4
import os

# Config 
MODEL_PATH = "weights/best.pt"   
MAX_MB = 8                       


app = FastAPI(title="VehDamage API", version="1.0", docs_url=None, redoc_url=None)

# CORS abierto para pruebas 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# carpeta para imágenes anotadas
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# cargar del modelo
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"No encuentro el modelo en {MODEL_PATH}. Pon tu best.pt ahí.")
model = YOLO(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/routes")
def routes():
    return [r.path for r in app.routes]

# Núcleo de inferencia 
def _infer(raw: bytes, conf: float):
    """
    Devuelve (detections:list, annotated_image:np.ndarray).
    """
    img = Image.open(BytesIO(raw)).convert("RGB")
    r = model.predict(img, conf=conf, verbose=False)[0]

    dets = []
    names = r.names if hasattr(r, "names") else {}
    has_masks = getattr(r, "masks", None) is not None and r.masks is not None

    n = len(r.boxes) if r.boxes is not None else 0
    for i in range(n):
        b = r.boxes[i]
        xyxy = [float(v) for v in b.xyxy[0].tolist()]
        cls = int(b.cls[0].item())
        confb = float(b.conf[0].item())
        item = {
            "bbox_xyxy": xyxy,
            "class_id": cls,
            "class_name": names.get(cls, str(cls)),
            "confidence": confb
        }
        # agregar polígono
        if has_masks and i < len(r.masks.xy):
            flat = []
            for x, y in r.masks.xy[i]:
                flat += [float(x), float(y)]
            item["polygon"] = flat
        dets.append(item)

    # imagen anotada (cajas + clase + prob; y máscaras)
    im_annot = r.plot()
    return dets, im_annot
# -----------------------------------------

# 1) JSON + URL relativa de la imagen anotada
@app.post("/api/predict")
async def api_predict(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0)
):
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=415, detail="Solo JPG/PNG.")
    raw = await file.read()
    if len(raw) > MAX_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Imagen > {MAX_MB} MB.")

    dets, im_annot = _infer(raw, conf)

    # guarda imagen anotada y devuelve una URL relativa para abrirla
    fname = f"{uuid4().hex}.jpg"
    Image.fromarray(im_annot).save(os.path.join("static", fname), "JPEG", quality=85)
    return JSONResponse({"detections": dets, "annotated_url": f"/static/{fname}"})

# 2) Imagen anotada directa (para ver/descargar sin copiar nada)
@app.post("/api/predict-image", response_class=Response)
async def api_predict_image(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0)
):
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=415, detail="Solo JPG/PNG.")
    raw = await file.read()
    if len(raw) > MAX_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Imagen > {MAX_MB} MB.")

    dets, im_annot = _infer(raw, conf)
    buf = BytesIO()
    Image.fromarray(im_annot).save(buf, "JPEG", quality=85)
    return Response(
        content=buf.getvalue(),
        media_type="image/jpeg",
        headers={"Content-Disposition": 'inline; filename="annotated.jpg"'}
    )

# 3) UI mínima para subir una foto y ver resultado (sin Swagger)
@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
<!doctype html><html><head><meta charset="utf-8" />
<title>VehDamage — UI</title>
<style>
 body{font-family:system-ui,Arial,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem}
 .card{border:1px solid #e5e7eb;border-radius:12px;padding:16px}
 button{padding:.6rem 1rem;border-radius:8px;border:1px solid #e5e7eb;background:#f3f4f6;cursor:pointer}
 img{max-width:100%;height:auto;border-radius:12px;border:1px solid #e5e7eb}
 .row{display:flex;gap:12px;align-items:center;margin:.75rem 0;flex-wrap:wrap}
 .muted{opacity:.75}
 #cols{display:grid;grid-template-columns:1fr;gap:16px}
 @media(min-width:900px){ #cols{grid-template-columns:3fr 2fr} }
</style></head><body>
<h1>VehDamage — Demo</h1>
<div class="card">
 <div class="row">
  <input id="file" type="file" accept="image/*" capture="environment" />
  <label>Confianza:
    <input id="conf" type="number" step="0.01" min="0" max="1" value="0.25" style="width:80px" />
  </label>
  <button id="run">Detectar daño</button>
  <span id="msg" class="muted"></span>
 </div>
 <div id="cols">
  <div id="image"></div>
  <div id="info">
   <h3>Detecciones</h3>
   <div id="hint" class="muted">Aún no hay resultados.</div>
   <ul id="list"></ul>
  </div>
 </div>
</div>
<script>
const $=id=>document.getElementById(id);
const file=$("file"), conf=$("conf"), run=$("run"), msg=$("msg");
const outImg=$("image"), list=$("list"), hint=$("hint");
run.onclick=async()=>{
  msg.textContent="";
  if(!file.files.length){ msg.textContent="Elige una imagen primero."; return; }
  try{
    // Imagen anotada (sin descargar JSON)
    const f1=new FormData(); f1.append("file", file.files[0]);
    const rImg=await fetch(`/api/predict-image?conf=${encodeURIComponent(conf.value)}`, {method:"POST", body:f1});
    if(!rImg.ok) throw new Error("Error imagen "+rImg.status);
    const blob=await rImg.blob();
    const url=URL.createObjectURL(blob);
    outImg.innerHTML=`<img src="${url}" alt="Resultado" />`;

    // JSON con detecciones (lista de clases/probs/polígonos)
    const f2=new FormData(); f2.append("file", file.files[0]);
    const rJson=await fetch(`/api/predict?conf=${encodeURIComponent(conf.value)}`, {method:"POST", body:f2});
    const data=await rJson.json();
    const dets=data.detections||[];
    list.innerHTML="";
    hint.textContent=dets.length? "": "Sin detecciones con el umbral elegido.";
    dets.forEach((d,i)=>{
      const p=Math.round((d.confidence||0)*1000)/10;
      const poly=d.polygon?`, polígonos: ${d.polygon.length/2} pts`:"";
      const li=document.createElement("li");
      li.textContent=`${i+1}. ${d.class_name??d.class_id} — ${p}%  [${d.bbox_xyxy.map(v=>v.toFixed(1)).join(", ")}${poly}]`;
      list.appendChild(li);
    });
  }catch(e){ msg.textContent="Hubo un problema: "+e.message; }
};
</script></body></html>
"""
