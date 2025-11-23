from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from models.schemas import MSTResponse
from services.logistics import generate_logistics_mst

app = FastAPI(
    title="Logistics Network API",
    description="API для анализа логистических сетей с использованием MST",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Дефолтный bbox для Казани
DEFAULT_BBOX = (48.8, 55.6, 49.3, 55.9)


@app.get("/", tags=["Root"])
def read_root():
    """Корневой эндпоинт с информацией об API"""
    return {
        "message": "Logistics Network API",
        "endpoints": {
            "GET /": "Информация об API",
            "POST /analyze": "Анализ логистической сети",
            "GET /map": "Получить HTML карту",
            "GET /mst": "Получить данные MST в JSON",
            "DELETE /cache": "Очистить кэш"
        }
    }


@app.post("/analyze", response_model=MSTResponse, tags=["Analysis"])
def analyze_logistics_network(
        west: float = Query(DEFAULT_BBOX[0], description="Западная долгота"),
        south: float = Query(DEFAULT_BBOX[1], description="Южная широта"),
        east: float = Query(DEFAULT_BBOX[2], description="Восточная долгота"),
        north: float = Query(DEFAULT_BBOX[3], description="Северная широта"),
        mode: str = Query("auto", description="Тип маршрута: Авто / Аэро / Морской / ЖД")
):
    """
    Анализ логистической сети и построение MST

    Возвращает JSON с данными о точках и рёбрах MST
    """
    bbox = (west, south, east, north)
    try:
        result = generate_logistics_mst(bbox, mode, cache_dir="cache")
        if result.get("status") != "ok":
            raise HTTPException(status_code=404, detail=result.get("message", "Ошибка анализа"))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/map", response_class=HTMLResponse, tags=["Visualization"])
def get_map(
        west: float = Query(DEFAULT_BBOX[0], description="Западная долгота"),
        south: float = Query(DEFAULT_BBOX[1], description="Южная широта"),
        east: float = Query(DEFAULT_BBOX[2], description="Восточная долгота"),
        north: float = Query(DEFAULT_BBOX[3], description="Северная широта"),
        mode: str = Query("auto")
):
    """
    Получить HTML карту с визуализацией MST

    Возвращает интерактивную карту Folium
    """
    bbox = (west, south, east, north)
    try:
        result = generate_logistics_mst(bbox, mode, cache_dir="results")
        if result.get("status") != "ok":
            return HTMLResponse(f"<h3>{result.get('message', 'Нет данных')}</h3>", status_code=404)

        html_path = result["map_path"]
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/cache", tags=["Cache"])
def clear_cache():
    """Очистить кэш OSM данных"""
    import os
    import shutil

    try:
        cache_dir = "cache"
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
            return {"message": "Кэш успешно очищен"}
        else:
            return {"message": "Кэш уже пуст"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", tags=["Health"])
def health_check():
    """Проверка здоровья API"""
    return {
        "status": "healthy",
        "service": "Logistics Network API"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
