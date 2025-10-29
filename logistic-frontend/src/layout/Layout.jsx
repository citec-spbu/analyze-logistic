import React, { useState } from "react";

export default function Layout({ children, sidebarContent }) {
    const [sidebarOpen, setSidebarOpen] = useState(false);

    return (
        <div style={{ position: "relative", width: "100%", height: "100vh", overflow: "hidden" }}>
            {/* Верхний блок / header */}
            <header
                style={{
                    position: "fixed",
                    top: 0,
                    left: 0,
                    width: "100%",
                    padding: "12px",
                    backgroundColor: "rgba(240, 240, 240, 0.5)",
                    textAlign: "center",
                    fontWeight: "bold",
                    zIndex: 1000,
                }}
            >
                🌍 Сервис по анализу региональной
                транспортно-логистической
                системы

            </header>

            {/* Сайдбар */}
            <div
                style={{
                    position: "fixed",
                    top: 0,
                    left: sidebarOpen ? "0" : "-340px",
                    width: "300px",
                    height: "100vh",
                    backgroundColor: "rgba(255, 255, 255, 0.9)",
                    boxShadow: sidebarOpen ? "2px 0 10px rgba(0,0,0,0.3)" : "none",
                    overflow: "hidden",
                    transition: "left 0.35s ease-in-out",
                    padding: "20px",
                    zIndex: 2000,
                    display: "flex",
                    flexDirection: "column",
                }}
            >
                <button
                    onClick={() => setSidebarOpen(false)}
                    style={{
                        alignSelf: "flex-end",
                        background: "none",
                        border: "none",
                        fontSize: "18px",
                        cursor: "pointer",
                        color: "#333",
                        marginBottom: "10px",
                    }}
                >
                    ✕
                </button>

                {/* Вставляем содержимое из пропа */}
                {sidebarContent}
            </div>

            {/* Кнопка открытия сайдбара */}
            {
                !sidebarOpen && (
                    <button
                        onClick={() => setSidebarOpen(true)}
                        style={{
                            position: "fixed",
                            top: "70px",
                            left: "20px",
                            zIndex: 1500,
                            padding: "10px 16px",
                            backgroundColor: "rgba(255,255,255,0.7)",
                            border: "1px solid #ccc",
                            borderRadius: "8px",
                            cursor: "pointer",
                            boxShadow: "0 2px 6px rgba(0,0,0,0.5)",
                        }}
                    >
                        ☰ Меню
                    </button>
                )
            }

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
                    opacity: 0,
                    transition: "opacity 0.3s",
                    zIndex: 1000,
                }}
                onMouseEnter={(e) => (e.currentTarget.style.opacity = 1)}
                onMouseLeave={(e) => (e.currentTarget.style.opacity = 0)}
            >
                <div style={{ fontSize: "12px", color: "#444" }}>
                    Карта © <a href="https://www.openstreetmap.org/copyright"
                        target="_blank"
                        rel="noopener noreferrer">OpenStreetMap</a> contributors
                    <br />
                    <a href="https://github.com/agalikeev/analyze-logistic" target="_blank" rel="noopener noreferrer">
                        Ссылка на наш github
                    </a>
                </div>
            </footer>
        </div >
    );
}
