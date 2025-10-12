import React from "react";

export default function Layout({ children }) {
    return (
        <div style={{ position: "relative", width: "100%", height: "100vh" }}>
            {/* Верхний блок / header */}
            <header
                style={{
                    position: "fixed",
                    top: 0,
                    left: 0,
                    width: "100%",
                    padding: "12px",
                    backgroundColor: "rgba(240, 240, 240, 0.8)",
                    textAlign: "center",
                    fontWeight: "bold",
                    zIndex: 1000,
                }}
            >
                🌍 Проект по Анализу Карт
            </header>

            {/* Основная рабочая область */}
            <main style={{ width: "100%", height: "100%" }}>
                {children}
            </main>

            {/* Нижний блок / footer */}
            <footer
                style={{
                    position: "fixed",
                    bottom: 0,
                    left: 0,
                    width: "100%",
                    padding: "8px",
                    textAlign: "center",
                    backgroundColor: "rgba(240,240,240,0.9)",
                    opacity: 0, // изначально прозрачный
                    transition: "opacity 0.3s",
                    zIndex: 1000,
                }}
                onMouseEnter={(e) => (e.currentTarget.style.opacity = 1)}
                onMouseLeave={(e) => (e.currentTarget.style.opacity = 0)}
            >
                <a href="https://github.com/agalikeev/analyze-logistic" target="_blank" rel="noopener noreferrer">
                    Ссылка на проект
                </a>
            </footer>
        </div>
    );
}
