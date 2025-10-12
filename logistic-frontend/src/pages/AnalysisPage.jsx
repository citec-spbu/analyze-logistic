import React, { useState } from "react";
import MapView from "../components/MapView";

export default function AnalysisPage() {
    // Состояние выбранной области
    const [selectedArea, setSelectedArea] = useState(null);

    function areaToGeoJSON(area) {
        return {
            type: "Feature",
            geometry: {
                type: "Polygon",
                coordinates: [[
                    [area.minLng, area.minLat],
                    [area.maxLng, area.minLat],
                    [area.maxLng, area.maxLat],
                    [area.minLng, area.maxLat],
                    [area.minLng, area.minLat],
                ]]
            },
            properties: {
                zoom: area.zoom || null,
            }
        };
    }

    async function sendGeoJSON(geojson) {
        try {
            const response = await fetch("https://httpbin.org/post", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(geojson)
            });
            const data = await response.json();
            console.log("🔁 Ответ от сервера:", data.json);
            alert("✅ Участок отправлен! Проверь консоль для деталей.");
        } catch (err) {
            console.error("❌ Ошибка при отправке:", err);
            alert("Ошибка при отправке данных на сервер!");
        }
    }

    // Кнопка "Начать Анализ"
    const handleAnalyze = async () => {
        if (!selectedArea) {
            alert("Выделите участок карты для анализа!");
            return;
        }

        // TODO: Логика обработки выделенного участка карты
        // Создаём GeoJSON-объект из выбранной области
        const geojson = areaToGeoJSON(selectedArea);
        console.log("Отправляем GeoJSON:", geojson);
        await sendGeoJSON(geojson);
    };


    return (
        <div style={{ position: "relative", flex: 1, width: "100%", height: "100%", minHeight: 0 }}>

            {/* Карта */}
            <MapView onAreaSelect={setSelectedArea} />

            {/* Кнопка поверх карты */}
            {selectedArea && (
                <button
                    onClick={handleAnalyze}
                    style={{
                        position: "absolute",
                        bottom: "40px",
                        left: "50%",
                        transform: "translateX(-50%)",
                        backgroundColor: "#0f62fe",
                        color: "white",
                        padding: "10px 20px",
                        border: "none",
                        borderRadius: "8px",
                        cursor: "pointer",
                        fontWeight: "bold",
                        boxShadow: "0 2px 6px rgba(0,0,0,0.2)",
                        zIndex: 1000,
                    }}
                >
                    Начать анализ
                </button>
            )}

            {/* Информация о выбранной области */}
            {selectedArea && (
                <div style={{
                    position: "absolute",
                    top: "60px",
                    left: "50%",
                    transform: "translateX(-50%)",
                    backgroundColor: "rgba(255, 255, 255, 0.9)",
                    padding: "6px 12px",
                    borderRadius: "6px",
                    zIndex: 1000,
                    fontWeight: "bold",
                }}>
                    ✅ Область выбрана
                </div>
            )}
        </div>
    );

}