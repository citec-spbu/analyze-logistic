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

```markdown
.
├── cache                   # Папка для результатов
├── logistic-backend/       # Бэкенд на FastAPI
│   ├── src                 # Основная папка
│   └── ...                 # Модули обработки данных, индексов, моделей
│
├── logistic-frontend/      # Фронтенд на Vite 
│   ├── package.json
│   ├── src/
│   └── ...
│
├── logistic-backend-python/         
│   ├── modeles
│   ├── services
│   ├── main.py             # исполняемый backend файл
│   └── ...
│
├── start.sh                # Скрипт запуска (Linux/macOS)
├── start.bat               # Скрипт запуска (Windows)
└── README.md
```


