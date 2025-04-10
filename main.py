from fastapi import FastAPI
import cosine_module 
import warnings
import json

app = FastAPI()
cosine_sim = cosine_module.CosineSimilarity()
warnings.filterwarnings("ignore", category=FutureWarning)

@app.get("/")
async def root():
    return {"message": "API de Similitud de Cosenos"}

@app.post("/similitud-cosenos")
async def rag_paradigm(texto1: str, texto2: str):
    try:
        result = cosine_sim.get_cosine_similarity(texto1, texto2)
        response = {
            "texto 1": texto1,
            "texto 2": texto2,
            "respuesta": result
        }
        return response
    except Exception as e:
        return {"error": f"Error al calcular similitud: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)  # Cambiado a 8080


