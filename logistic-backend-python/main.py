from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from models.schemas import BoundingBox, MSTResponse, MapResponse
from services.logistics import LogisticsService
from typing import Optional
import json

app = FastAPI(
    title="Logistics Network API",
    description="API для анализа логистических сетей с использованием MST",
    version="1.0.0"
)

# Инициализация сервиса
logistics_service = LogisticsService()

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
        north: float = Query(DEFAULT_BBOX[3], description="Северная широта")
):
    """
    Анализ логистической сети и построение MST

    Возвращает JSON с данными о точках и рёбрах MST
    """
    try:
        bbox = (west, south, east, north)

        # Загрузка данных
        centers_gdf = logistics_service.load_logistics_centers(bbox)

        if centers_gdf.empty:
            raise HTTPException(
                status_code=404,
                detail="Не найдено логистических центров в заданной области"
            )

        # Извлечение координат
        coords_df = logistics_service.extract_coordinates(centers_gdf)

        # Построение графа и MST
        G = logistics_service.build_graph(coords_df)
        mst = logistics_service.calculate_mst(G)

        # Получение данных
        mst_data = logistics_service.get_mst_data(coords_df, mst)

        return MSTResponse(**mst_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/map", response_class=HTMLResponse, tags=["Visualization"])
def get_map(
        west: float = Query(DEFAULT_BBOX[0], description="Западная долгота"),
        south: float = Query(DEFAULT_BBOX[1], description="Южная широта"),
        east: float = Query(DEFAULT_BBOX[2], description="Восточная долгота"),
        north: float = Query(DEFAULT_BBOX[3], description="Северная широта")
):
    """
    Получить HTML карту с визуализацией MST

    Возвращает интерактивную карту Folium
    """
    try:
        bbox = (west, south, east, north)

        # Загрузка данных
        centers_gdf = logistics_service.load_logistics_centers(bbox)

        if centers_gdf.empty:
            return HTMLResponse(
                content="<h1>Не найдено логистических центров в заданной области</h1>",
                status_code=404
            )

        # Извлечение координат
        coords_df = logistics_service.extract_coordinates(centers_gdf)

        # Построение графа и MST
        G = logistics_service.build_graph(coords_df)
        mst = logistics_service.calculate_mst(G)

        # Создание карты
        m = logistics_service.create_map(coords_df, mst, bbox)

        return HTMLResponse(content=m._repr_html_())

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mst", response_model=MSTResponse, tags=["Analysis"])
def get_mst_data(
        west: float = Query(DEFAULT_BBOX[0], description="Западная долгота"),
        south: float = Query(DEFAULT_BBOX[1], description="Южная широта"),
        east: float = Query(DEFAULT_BBOX[2], description="Восточная долгота"),
        north: float = Query(DEFAULT_BBOX[3], description="Северная широта")
):
    """
    Получить данные MST в формате JSON

    Альтернатива /analyze с тем же функционалом
    """
    return analyze_logistics_network(west, south, east, north)


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