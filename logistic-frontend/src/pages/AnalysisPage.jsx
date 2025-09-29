import React, { useState } from "react";

export default function AnalysisPage() {
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);

    // Выбор файла
    const handleFileChange = (event) => {
        const selectedFile = event.target.files[0];
        if (!selectedFile) return;

        setFile(selectedFile);
        const reader = new FileReader(); // Читаем файт в памяти браузера
        reader.onload = (e) => setPreview(e.target.result); // Сохраняем результат в Preview
        reader.readAsDataURL(selectedFile); // Преобразуем файл  в строку
    };

    // Кнопка "Начать Анализ"
    const handleAnalyze = () => {
        if (!file) {
            alert("Выберите файл для карты!");
            return;
        }

        // TODO: Вызов API для анализа
        console.log("Отправка файла на сервер:", file);

        alert("Файл отправлен на сервер");
    };


    return (
        <div>
            <h1>Анализ Карты</h1>
            <p> Карта и визуализация Графа</p>

        <div style={{ marginBottom: 16}}>
            <input type="file" accept="image/*" onChange={handleFileChange}/>
            <button onClick={handleAnalyze} style={{ marginLeft: 8}}>
                Начать Анализ
            </button>
        </div>

        {file && (
            <div>
                <p>Выбран файл: {file.name}</p>
                {preview && (
                    <img
                    src={preview}
                    alt="Превью карты"
                    style={{ maxWidth: "100%", maxHeight: 300, border: "1px solid #ccc"}} />
                )}
            </div>
        )}
        </div>
    );
}