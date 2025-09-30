import React from "react";
import { MapContainer, TileLayer } from "react-leaflet";

// По умолчанию стоят координаты ПМПУ
export default function MapView({ center = [59.882036, 29.829662], zoom = 17 }) {
    return (
        <div style={{ width: "100%", height: "500px" }}>
            <MapContainer center={center} zoom={zoom} style={{ width: "100%", height: "100%" }}>
                <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                    attribution='&copy; OpenStreetMap contributors' />
            </MapContainer>
        </div>
    );
}