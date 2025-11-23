# Сервис по анализу регионально-транспортной логистической системы

---

## Инструкция по запуску

### Требования
- [Node.js](https://nodejs.org/) v18+
- [npm](https://www.npmjs.com/) (входит в состав Node.js)
- [Python](https://www.python.org/) 3.9+

### Установка и запуск
Установка зависимостей
```bash
cd logistic-backend-python/ && pip install -r "requirements.txt" && cd ..
```
Запуск приложения Windows

```bash
start.bat
```

Запуск приложения Linux

```bash
./start.sh
```

### Структура репозитория

.
├── cache                   # Папка для результатов
├── logistic-backend/       # Бэкенд на FastAPI
│   ├── app.py              # Основной файл приложения
│   ├── requirements.txt    # Зависимости Python
│   └── ...                 # Модули обработки данных, индексов, моделей
│
├── logistic-frontend/      # Фронтенд на Vite + React
│   ├── package.json
│   ├── src/
│   └── ...
│
├── logistic-backend-python/         
│   ├── package.json
│   ├── src/
│   └── ...
│
├── start.sh                # Скрипт запуска (Linux/macOS)
├── start.bat               # Скрипт запуска (Windows)
└── README.md



