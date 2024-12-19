import streamlit as st

import os
import ast

from pathlib import Path
import numpy as np
import pandas as pd
import gdown

# import random
import requests
from dotenv import load_dotenv

import tensorflow as tf
from tensorflow.keras.models import load_model

import speech_recognition as sr

# from utils import *


# Load environment variables from .env file
load_dotenv()

# Access the API credentials
EBIRD_API_URL = os.getenv("EBIRD_API_URL")
EBIRD_API_KEY = os.getenv("EBIRD_API_KEY")
WIKIMEDIA_API_URL = os.getenv("WIKIMEDIA_API_URL")


# Load data from CSV
@st.cache_data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        if "type" in data.columns:
            data["type"] = data["type"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        return data
    except FileNotFoundError:
        st.error(f"Файл {file_path} не найден. Проверьте путь и повторите попытку.")
        return pd.DataFrame()


# Retrieve bird info via eBird API
# @st.cache_data
def get_bird_info(species_code):
    url = f"{EBIRD_API_URL}?species={species_code}&fmt=json&locale=ru"
    headers = {"X-eBirdApiToken": EBIRD_API_KEY}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Не удалось получить информацию о птице. Статус: {response.status_code}")
    except Exception as e:
        st.error(f"Ошибка при запросе данных: {e}")
    return None


# Retrieve bird image via Wikimedia API
def get_bird_image(bird_name):
    formatted_bird_name = bird_name.replace(" ", "_").lower()
    params = {
        "action": "query",
        "format": "json",
        "prop": "pageimages",
        "titles": formatted_bird_name,
        "pithumbsize": 500
    }
    try:
        response = requests.get(WIKIMEDIA_API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            if "thumbnail" in page_data:
                return page_data["thumbnail"]["source"]
            else:
                return None
    except requests.RequestException:
        return None
    except Exception:
        return None


# Function for tracking the dynamics
def bird_dynamics(df, bird='', longitude_left=-180, longitude_right=180, latitude_min=-90, latitude_max=90,
                  start_date=None, end_date=None, selected_seasons=None):
    """
    Функция выдает датафрейм с динамикой количества записей конкретного вида птиц с учетом фильтров по локации и периоду.
    Возвращает стилизованный датафрейм, где строки закрашены в цвет уровня риска.
    """
    # Проверка необходимых параметров
    if not bird:
        print('Необходимо указать код птицы (primary_label)!')
        return None

    # Преобразование дат
    if start_date:
        start_date = pd.to_datetime(start_date)
    if end_date:
        end_date = pd.to_datetime(end_date)

    # Фильтрация по дате
    if start_date or end_date:
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        if start_date:
            df = df[df['date'] >= start_date]
        if end_date:
            df = df[df['date'] <= end_date]

    # Фильтрация по сезонам
    if selected_seasons:
        df = df[df['season'].isin(selected_seasons)]

    # Проверка наличия данных после фильтрации
    if df.empty:
        print('Данные с указанными параметрами не обнаружены')
        return None

    # Фильтрация по локации
    df_filtered = df[(df['longitude'] >= longitude_left) & (df['longitude'] <= longitude_right) &
                     (df['latitude'] >= latitude_min) & (df['latitude'] <= latitude_max)]

    # Группировка по годам для всех видов
    df_total_records = df_filtered.groupby(df_filtered['date'].dt.year).agg({'latitude': 'count'}).reset_index()
    df_total_records.columns = ["Год", "Общее количество записей"]

    # Фильтрация данных для конкретного вида
    df_bird = df_filtered[df_filtered['primary_label'] == bird]
    df_bird_records = df_bird.groupby(df_bird['date'].dt.year).agg({'latitude': 'count'}).reset_index()
    df_bird_records.columns = ["Год", "Количество записей вида"]

    # Объединение данных
    df_result = pd.merge(df_total_records, df_bird_records, on='Год', how='left').fillna(0)
    df_result['Количество записей вида'] = df_result['Количество записей вида'].astype(int)

    # Рассчёт относительной частоты записей (единиц на тысячу записей)
    df_result['Частота'] = df_result['Количество записей вида'] / df_result['Общее количество записей'] * 1000

    # Скользящее среднее и риск вымирания
    min_records_in_database = 40
    min_mov_avg_or_bird_counts = 3
    threshold_drop_in_counts = 0.7
    threshold_drop_in_frequency = 0.8

    df_result['Скользящее среднее (всего)'] = df_result['Общее количество записей'].rolling(3, min_periods=1).mean()
    df_result['Скользящее среднее (вида)'] = df_result['Количество записей вида'].rolling(3, min_periods=1).mean()
    df_result['Скользящее среднее (частота)'] = df_result['Частота'].rolling(3, min_periods=1).mean()

    df_result['Риск вымирания'] = 'Нет данных'
    for i in range(len(df_result)):
        if (df_result.loc[i, 'Скользящее среднее (всего)'] >= min_records_in_database) and \
           (df_result.loc[i, 'Скользящее среднее (вида)'] >= min_mov_avg_or_bird_counts):
            if (df_result.loc[i, 'Количество записей вида'] / df_result.loc[i, 'Скользящее среднее (вида)'] <= threshold_drop_in_counts) or \
               (df_result.loc[i, 'Частота'] / df_result.loc[i, 'Скользящее среднее (частота)'] <= threshold_drop_in_frequency):
                if (df_result.loc[i, 'Количество записей вида'] / df_result.loc[i, 'Скользящее среднее (вида)'] <= threshold_drop_in_counts) and \
                   (df_result.loc[i, 'Частота'] / df_result.loc[i, 'Скользящее среднее (частота)'] <= threshold_drop_in_frequency):
                    df_result.loc[i, 'Риск вымирания'] = 'Высокий'
                else:
                    df_result.loc[i, 'Риск вымирания'] = 'Средний'
            else:
                df_result.loc[i, 'Риск вымирания'] = 'Низкий'

    return df_result


###
### ML PART
###


BATCH_SIZE = 64
CLASSES_DICT = {0: 'england', 1: 'us'}


###==============================================


def get_models_list(dir_name, exts=('h5', 'joblib')):
    return [dir_name + f for f in os.listdir(dir_name) 
            if os.path.isfile(os.path.join(dir_name, f)) 
            and f.split('.')[-1] in exts]


class Info(tf.keras.layers.Layer):
    def __init__(self, classes_dict):
        self.classes_dict = classes_dict
        super().__init__()

    def get_config(self):
        return {'classes_dict': self.classes_dict}


def get_classes_dict(model):
    d = next(l.classes_dict for l in model.layers 
             if hasattr(l, 'classes_dict'))
    return {int(k): v for k, v in d.items()}


@st.cache_resource
def load_models(models_urls, models_dir='models'):
    '''Load pretrained models'''
    # Open a new TensorFlow session
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    session = tf.compat.v1.Session(config=config)
    with session.as_default():
        models = {}
        save_dest = Path(models_dir)
        save_dest.mkdir(exist_ok=True)

        # for m in get_models_list(models_dir):
        for i, m_url in enumerate(models_urls):
            model_file = os.path.join(models_dir, f'model_{i}.h5')
            if not Path(model_file).exists():
                with st.spinner("Downloading model... this may take a while! \n Don't stop!"):
                    # Download the file from Google Drive
                    gdown.download(m_url, model_file, quiet=False)

            if os.path.splitext(model_file)[-1] == '.h5':
                try:
                    model = load_model(model_file, custom_objects={'Info': Info})
                except:
                    model = load_model(model_file)

            if not hasattr(model, 'classes_dict'):
                try:
                    model.classes_dict = get_classes_dict(model)
                except:
                    model.classes_dict = CLASSES_DICT

            models.update({model.name: model})
    return session, models


def make_prediction(model, features, batch_size):
    # predict = model.predict(features, batch_size)
    pred_proba = model.predict(features)
    # pred_proba = model.predict_proba(features, 1)
    pred = np.argmax(pred_proba, axis=-1)
    return pred_proba, pred


def recognize_class(model, features):
    # features = np.expand_dims(features, 0)
    pred_proba, pred = make_prediction(model, features, BATCH_SIZE)
    # pred = make_prediction(model, features, 1)[0]
    return pred_proba[0], model.classes_dict[pred[0]]
