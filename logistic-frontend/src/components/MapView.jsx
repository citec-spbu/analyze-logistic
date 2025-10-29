import React, { useState, useEffect } from "react";
import { MapContainer, Marker, Polygon, TileLayer, useMapEvents, ZoomControl } from "react-leaflet";

// Компонент выделения области на карте
function SelectArea({ onChange, point1, point2, setPoint1, setPoint2 }) {
    // Хранит координаты кликов
    const [points, setPoints] = useState([]);

    // При изменении точек из сайдбара ставим точки на карту
    useEffect(() => {
        if (!point1 || !point2) {
            setPoints([]);
            onChange(null);
            return;
        }

        const p1 = point1.split(",").map(Number);
        const p2 = point2.split(",").map(Number);
        if (p1.length === 2 && p2.length === 2) {
            setPoints([{ lat: p1[0], lng: p1[1] }, { lat: p2[0], lng: p2[1] }]);
            onChange({
                minLat: Math.min(p1[0], p2[0]),
                maxLat: Math.max(p1[0], p2[0]),
                minLng: Math.min(p1[1], p2[1]),
                maxLng: Math.max(p1[1], p2[1]),
            });
        }
    }, [point1, point2, onChange]);

    // Для обработки кликов по карте
    useMapEvents({
        click(e) {
            if (points.length < 2) {
                const newPoints = [...points, e.latlng];
                setPoints(newPoints);

                if (newPoints.length === 2) {
                    const [p1, p2] = newPoints;
                    const areaObject = {
                        minLat: Math.min(p1.lat, p2.lat),
                        maxLat: Math.max(p1.lat, p2.lat),
                        minLng: Math.min(p1.lng, p2.lng),
                        maxLng: Math.max(p1.lng, p2.lng),
                    };
                    onChange(areaObject);
                    setPoint1(`${p1.lat.toFixed(6)},${p1.lng.toFixed(6)}`);
                    setPoint2(`${p2.lat.toFixed(6)},${p2.lng.toFixed(6)}`);
                }
            } else {
                setPoints([e.latlng]);
                onChange(null);
                setPoint1("");
                setPoint2("");
            }
        }

    });

    return (
        <>
            {/* Рендер маркеров */}
            {points.map((p, idx) => <Marker key={idx} position={p} />)}

            {/* Рисуем прямоугольник */}
            {points.length === 2 && (
                <Polygon
                    positions={[
                        [points[0].lat, points[0].lng],
                        [points[0].lat, points[1].lng],
                        [points[1].lat, points[1].lng],
                        [points[1].lat, points[0].lng],
                        [points[0].lat, points[0].lng],
                    ]}
                />
            )}
        </>
    );
}

// По умолчанию стоят координаты ПМПУ
export default function MapView({ center = [59.882036, 29.829662], zoom = 17, onAreaSelect, point1, point2, setPoint1, setPoint2 }) {
    return (
        <MapContainer
            center={center}
            zoom={zoom}
            style={{
                width: "100%",
                height: "100vh",
            }}
            zoomControl={false}
        >
            <TileLayer
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                attribution=''
            />

            {/* Компонент выделения области */}
            <SelectArea
                onChange={onAreaSelect}
                point1={point1}
                point2={point2}
                setPoint1={setPoint1}
                setPoint2={setPoint2}
            />
        </MapContainer>
    );
}