# Hackathon 1. Team 3

## Описание

**MVP** ML-driven продукта на основе готовых предобученных моделей машинного обучения.  

**Тематика:**  
*Мониторинг биоразнообразия редких видов птиц на основе аудиозаписей их пения.*

---

## Архитектура

**Файловая структура:**
```plaintext
project_root/
│-- README.md
│-- [Hackathon_1]_BirdCLEF_EDA_and_Preprocessing.ipynb     # Ноутбук с проведённым EDA и предобработкой данных
│-- [Hackathon_1]_BirdCLEF_Model_Training-Effnetv2.ipynb   # Ноутбук с полным процессом дообучения модели EfficientNetV2-B2
│-- dataset/                                               # Директория с датасетами 2021 и 2024 гг.
│-- app/                                                   # Директория с кодом приложения
    │-- app.py                                             # Основной скрипт Streamlit приложения
    │-- ui_classify.py                                     # Интерфейс классификации видов
    │-- ui_visualize.py                                    # Интерфейс визуализации и аналитики
    │-- core_.py                                           # Логика обработки данных и ML
    │-- requirements.txt                                   # Зависимости проекта
    │-- to_predict__train.csv                              # Файл метаданных аудиозаписей, на которых была обучена модель
    │-- models/                                            # Директория для хранения моделей
```

**Технологический стек:**  
- python, streamlit, tensorflow, pandas, numpy, pydeck, scikit-image, imgviz, librosa, pydub

## Установка и запуск

**Пререквизиты:** Python 3.8+, pip

1. **Клонировать репозиторий**:
```bash
git clone <..>
cd app
```

2. **Установить библиотеки:**
```bash
pip install -r requirements.txt
```

3. **Запусить демо-приложение на Streamlit:**
```bash
streamlit run app.py
```

---

## Команда проекта

|                      | Участник             | Роли и задачи                             |
|-----------------------|------------------|-------------------------------------------|
| **Teamlead**         | Шумилин Антон    | ML Engineering, Data Preprocessing, Streamlit |
| **Data Engineer**     | Николай Аушкап   | EDA, ML Engineering |
| **Data Engineer**     | Мартынов Алексей | EDA, Data Proprocessing |
| **Data Analyst**      | Фахретдинов Муса | EDA, Data Proprocessing |
| **Frontend Dev**      | Лаврухина Виктория | Streamlit, Viz, API, EDA |
| **Data Analyst**      | Фивейский Сергей | Data Preprocessing |
