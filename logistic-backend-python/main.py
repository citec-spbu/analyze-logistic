from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


from models.schemas import MSTResponse
from services.logistics import generate_logistics_mst,  analyze_logistics_metrics, generate_all_modes_mst
from services.logistics import compute_metric
import os

app = FastAPI(
    title="Logistics Network API",
    description="API для анализа логистических сетей с использованием MST",
    version="1.0.0"
)
app.mount("/cache", StaticFiles(directory="cache"), name="cache")

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


@app.post("/analyze", tags=["Analysis"])
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


@app.get("/map/all", response_class=HTMLResponse, tags=["Visualization"])
def get_map_all(
        west: float = Query(DEFAULT_BBOX[0]),
        south: float = Query(DEFAULT_BBOX[1]),
        east: float = Query(DEFAULT_BBOX[2]),
        north: float = Query(DEFAULT_BBOX[3]),
):
    """
    Получить карту MST для всех модов одновременно
    """
    bbox = (west, south, east, north)
    try:
        html_path = generate_all_modes_mst(bbox, cache_dir="cache")
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
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
    """
    bbox = (west, south, east, north)
    try:
        result = generate_logistics_mst(bbox, mode, cache_dir="cache")
        if result.get("status") != "ok":
            return HTMLResponse(f"<h3>{result.get('message', 'Нет данных')}</h3>", status_code=404)

        html_path = result["map_path"]
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/metrics", tags=["Metrics"])
def calculate_metrics(
        metric: str = Query(..., description="Тип метрики: degree_centrality / closeness_centrality / betweenness_centrality / pagerank"),
        west: float = Query(DEFAULT_BBOX[0]),
        south: float = Query(DEFAULT_BBOX[1]),
        east: float = Query(DEFAULT_BBOX[2]),
        north: float = Query(DEFAULT_BBOX[3]),
        mode: str = Query("auto")
):
    """
    Анализ графовой метрики и генерация HTML-карты результата.
    """
    bbox = (west, south, east, north)

    try:
        result = analyze_logistics_metrics(bbox=bbox, mode=mode, metric=metric,cache_dir="cache")

        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result.get("message", "Ошибка метрики"))

        if not os.path.exists(result["map_path"]):
            raise HTTPException(
                status_code=500,
                detail=f"Файл карты не создан: {result['map_path']}"
            )
        return {
            "map_path": result["map_path"],
            "metric": metric,
            "status": "ok"
        }

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