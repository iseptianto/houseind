# File: app.py

from fastapi import FastAPI
import uvicorn

# Inisiasi Aplikasi FastAPI
app = FastAPI()

# Endpoint Test Pertama
@app.get("/")
def read_root():
    # Ini akan muncul di browser: {"status": "success", "service": "FastAPI API"}
    return {"status": "success", "service": "FastAPI API"}

# Endpoint Test Kedua (untuk testing model, nanti)
@app.get("/status")
def get_status():
    return {"status": "ok", "version": "1.0"}

# Ini penting untuk memastikan uvicorn terinstal dan dapat dijalankan
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
