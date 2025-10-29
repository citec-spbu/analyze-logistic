import React, { useState } from "react";
import Layout from "../layout/Layout";
import MapView from "../components/MapView";

export default function AnalysisPage() {
    // Состояние выбранной области
    const [selectedArea, setSelectedArea] = useState(null);
    const [point1, setPoint1] = useState("");
    const [point2, setPoint2] = useState("");
    const [selectedMode, setSelectedMode] = useState("auto");

    const sidebarContent = (
        <>
            <h3 style={{ marginTop: 0, marginBottom: "16px" }}>📍 Ввод координат</h3>
            <label>
                Точка 1 (lat,lng):
                <input
                    type="text"
                    placeholder="59.88,29.82"
                    value={point1}
                    onChange={(e) => setPoint1(e.target.value)}
                    style={{ width: "100%", marginTop: "4px", padding: "6px 8px", borderRadius: "6px", border: "1px solid #ccc" }}
                />
            </label>
            <label>
                Точка 2 (lat,lng):
                <input
                    type="text"
                    placeholder="59.88,29.83"
                    value={point2}
                    onChange={(e) => setPoint2(e.target.value)}
                    style={{ width: "100%", marginTop: "4px", padding: "6px 8px", borderRadius: "6px", border: "1px solid #ccc" }}
                />
            </label>

            {/* Блок выбора режима анализа */}
            <div style={{ marginTop: "20px" }}>
                <h4 style={{ marginBottom: "8px" }}>🚦 Тип маршрута:</h4>

                <label style={{ display: "block", marginBottom: "6px" }}>
                    <input
                        type="radio"
                        name="mode"
                        value="auto"
                        checked={selectedMode === "auto"}
                        onChange={() => setSelectedMode("auto")}
                    />{" "}
                    Автомобильный
                </label>

                <label style={{ display: "block", marginBottom: "6px" }}>
                    <input
                        type="radio"
                        name="mode"
                        value="aero"
                        checked={selectedMode === "aero"}
                        onChange={() => setSelectedMode("aero")}
                    />{" "}
                    Авиамаршрут
                </label>

                <label style={{ display: "block", marginBottom: "6px" }}>
                    <input
                        type="radio"
                        name="mode"
                        value="sea"
                        checked={selectedMode === "sea"}
                        onChange={() => setSelectedMode("sea")}
                    />{" "}
                    Морской маршрут
                </label>

                <label style={{ display: "block" }}>
                    <input
                        type="radio"
                        name="mode"
                        value="rail"
                        checked={selectedMode === "rail"}
                        onChange={() => setSelectedMode("rail")}
                    />{" "}
                    Железнодорожный
                </label>
            </div>
        </>
    );

    function areaToGeoJSON(area,routemode) {
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
                mode: routemode || "auto",
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
        const geojson = areaToGeoJSON(selectedArea, selectedMode);
        console.log("Отправляем GeoJSON:", geojson);
        await sendGeoJSON(geojson);
    };


    return (
        <div style={{ position: "relative", flex: 1, width: "100%", height: "100%", minHeight: 0 }}>

            <Layout sidebarContent={sidebarContent}>

                {/* Карта */}
                <MapView
                    onAreaSelect={setSelectedArea}
                    point1={point1}
                    point2={point2}
                    setPoint1={setPoint1}
                    setPoint2={setPoint2}
                />
            </Layout>

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