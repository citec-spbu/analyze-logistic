import React, { useState } from "react";
import { MapContainer, Marker, Polygon, TileLayer, useMapEvents, ZoomControl } from "react-leaflet";

// Компонент выделения области на карте
function SelectArea({ onChange }) {
    // points хранит координаты кликов
    const [points, setPoints] = useState([]);

    // Для обработки кликов по карте
    useMapEvents({
        click(e) {
            if (points.length < 2) {
                // Добавление новой точки
                const newPoints = [...points, e.latlng];
                setPoints(newPoints);

                if (newPoints.length === 2) {
                    // callback с координатами прямоугольника
                    const [p1, p2] = newPoints;
                    const minLat = Math.min(p1.lat, p2.lat);
                    const maxLat = Math.max(p1.lat, p2.lat);
                    const minLng = Math.min(p1.lng, p2.lng);
                    const maxLng = Math.max(p1.lng, p2.lng);
                    const areaObject = { minLat, minLng, maxLat, maxLng };
                    onChange(areaObject);
                }
            } else {
                // Сброс при третьем клике
                setPoints([e.latlng]);
                onChange(null);
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
                        [Math.min(points[0].lat, points[1].lat), Math.min(points[0].lng, points[1].lng)],
                        [Math.min(points[0].lat, points[1].lat), Math.max(points[0].lng, points[1].lng)],
                        [Math.max(points[0].lat, points[1].lat), Math.max(points[0].lng, points[1].lng)],
                        [Math.max(points[0].lat, points[1].lat), Math.min(points[0].lng, points[1].lng)],
                    ]}
                />
            )}
        </>
    );
}

// По умолчанию стоят координаты ПМПУ
export default function MapView({ center = [59.882036, 29.829662], zoom = 17, onAreaSelect }) {
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
                attribution='&copy; OpenStreetMap contributors'
            />
            {/* Компонент выделения области */}
            <SelectArea onChange={onAreaSelect} />
        </MapContainer>
    );
}