from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/routes")
def routes():
    return [r.path for r in app.routes]

@app.get("/ui", response_class=HTMLResponse)
def ui():
    return "<h1>UI OK ✅</h1><p>Si ves esto, FastAPI está bien. Luego cargamos el modelo.</p>"