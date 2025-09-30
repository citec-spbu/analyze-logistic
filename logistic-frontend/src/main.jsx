// Импорт React и функции createRoot
import React from "react";
import { createRoot } from "react-dom/client";

// Подключение приложения
import App from "./App";

// Подключение стилей
import "./index.css";
import "leaflet/dist/leaflet.css";
import "./leafletIcons";

createRoot(document.getElementById("root")).render(
 <React.StrictMode>
    <App />
  </React.StrictMode>
);
