# 1) Лёгкий образ с Python
FROM python:3.10-slim

# 2) Рабочая директория внутри контейнера
WORKDIR /app

# 3) Ставим только Python-зависимости
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# 4) Копируем весь код и данные (индекс, модель и т.д.)
COPY . .

# 5) Открываем порт Gradio
EXPOSE 7860

# 6) Команда по умолчанию
CMD ["python", "app.py"]
