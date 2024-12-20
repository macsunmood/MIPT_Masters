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

# import tensorflow as tf
# from tensorflow.keras.models import load_model

import librosa
from tqdm import tqdm
import tensorflow as tf
import tensorflow_hub as hub


import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# from utils import *


# Load environment variables from .env file
load_dotenv()

# Access the API credentials
EBIRD_API_URL = os.getenv("EBIRD_API_URL")
EBIRD_API_KEY = os.getenv("EBIRD_API_KEY")
WIKIMEDIA_API_URL = os.getenv("WIKIMEDIA_API_URL")

model = None
df_model_labels = pd.read_csv('data/labels.csv')


# Load data from CSV
@st.cache_data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        # if "type" in data.columns:
        #     data["type"] = data["type"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        # return data

        data['primary_label'] = data['primary_label'].astype(str)


        print(data['primary_label'].dtype)
        print(data['primary_label'].apply(type).value_counts())
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
# @st.cache_data
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


# Core function for tracking the dynamics
def bird_dynamics(df, bird='', longitude_left=-180, longitude_right=180, latitude_min=-90, latitude_max=90,
                  start_date=None, end_date=None, selected_seasons=None):
    """
    Функция выдает датафрейм с динамикой количества записей конкретного вида птиц с учетом фильтров по локации и периоду.
    Возвращает стилизованный датафрейм, где строки закрашены в цвет уровня риска.
    """
    # Проверка обязательных параметров
    if bird == '':
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

    # Служебные переменные и словари для дальнейшего анализа
    first_year = df_result['Год'].min()
    last_year = df_result['Год'].max()
    dict_total = df_result.set_index('Год')['Общее количество записей'].to_dict()
    dict_bird = df_result.set_index('Год')['Количество записей вида'].to_dict()

    # Заполнение в словаре данных по пропущенным годам
    for i in range(first_year, last_year + 1):
        if dict_total.get(i) is None:
            dict_total[i] = 0
            dict_bird[i] = 0

    # Создаем колонки для дальнейшего заполнения со значениями по умолчанию
    df_result['Скользящее среднее (всего)'] = df_result['Общее количество записей'].astype(float)
    df_result['Скользящее среднее (вида)'] = df_result['Количество записей вида'].astype(float)
    df_result['Скользящее среднее (частота)'] = df_result['Частота']
    df_result['Риск вымирания'] = 'Нет данных'

    # Заполняем колонки результирующего датафрейма (цикл по всем строкам)
    for i in range(df_result.shape[0]):
        if i < 2:  # для первых двух строк средневзвешенные не считаются
            df_result.loc[i, ['Скользящее среднее (всего)']] = 0
            df_result.loc[i, ['Скользящее среднее (вида)']] = 0
            df_result.loc[i, ['Скользящее среднее (частота)']] = 0
        else:
            year_number = df_result.loc[i]['Год']
            
            # Считаем средневзвешенные за 3 года
            df_result.loc[i, ['Скользящее среднее (всего)']] = (dict_total[year_number] + 
                                                                dict_total[year_number - 1] + 
                                                                dict_total[year_number - 2]) / 3
            ma_birds = (dict_bird[year_number] + 
                        dict_bird[year_number - 1] + 
                        dict_bird[year_number - 2]) / 3
            df_result.loc[i, ['Скользящее среднее (вида)']] = ma_birds
            ma_frequency = df_result.loc[i]['Скользящее среднее (вида)'] / df_result.loc[i]['Скользящее среднее (всего)'] * 1000
            df_result.loc[i, ['Скользящее среднее (частота)']] = ma_frequency
            
            # Оцениваем показатель риска - проверяем на пробитие порогов
            if (dict_total[year_number] >= min_records_in_database) and (ma_birds >= min_mov_avg_or_bird_counts):
                if dict_bird[year_number] / ma_birds <= threshold_drop_in_counts:
                    if df_result.loc[i]['Частота'] / ma_frequency <= threshold_drop_in_frequency:
                        df_result.loc[i, ['Риск вымирания']] = 'Высокий'
                    else:
                        df_result.loc[i, ['Риск вымирания']] = 'Средний'
                else:
                    if df_result.loc[i]['Частота'] / ma_frequency <= threshold_drop_in_frequency:
                        df_result.loc[i, ['Риск вымирания']] = 'Средний'
                    else:
                        df_result.loc[i, ['Риск вымирания']] = 'Низкий'

    return df_result


###
### ML PART
###


BATCH_SIZE = 64
CLASSES_DICT = {
    0: '0', 
    1: '1'
}


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
def load_model():
    model_url = 'https://kaggle.com/models/google/bird-vocalization-classifier/frameworks/tensorFlow2/variations/bird-vocalization-classifier/versions/4'
            # 'https://kaggle.com/models/google/bird-vocalization-classifier/TensorFlow2/bird-vocalization-classifier'
    global model
    model = hub.load(model_url)



@st.cache_resource
def load_models(models_urls, models_dir='models'):
    '''Load pretrained models'''
    # Open a new TensorFlow session
    # config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    # session = tf.compat.v1.Session(config=config)
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

def predict_species(filename):
    # функция меняет sample rate. Это нужно, так как модель обучена на 32000 Hz
    def resample_rate(audio, sample_rate, new_sample_rate=32000):
        if sample_rate != new_sample_rate:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=new_sample_rate)
        return audio, new_sample_rate

    # функция разбивает запись на фреймы по 5 секунд, так как модель обучена на работе с 5 секундными записями
    def break_into_frames(audio, sample_rate, window_size=5.0, step_size=5.0) -> np.ndarray:
        frame_length = int(window_size * sample_rate)
        frame_step  = int(step_size * sample_rate)
        framed_audio = tf.signal.frame(audio, frame_length, frame_step, pad_end=True)  # разбитое на фреймы аудио
        return framed_audio

    # получим предсказание для первого фрейма одной записи
    audio, sample_rate = librosa.load("audio/{filename}")

    audio, sample_rate = resample_rate(audio, sample_rate)
    framed_audio = break_into_frames(audio, sample_rate)  # разбитое на фреймы аудио

    logits, embeddings = model.infer_tf(framed_audio[0:1])  # получаем прогноз для первого фрейма
    embeddings = tf.nn.softmax(logits)

    # embeddings = model.infer_tf(framed_audio[0:1])['embedding']  # получаем прогноз для первого фрейма

    argmax = np.argmax(embeddings)
    birds_kind = df_model_labels.iloc[argmax]['ebird2021']
    probability = embeddings[0][argmax]
    print(f"Предсказанный вид птицы по первому фрейму: {birds_kind}, с вероятностью: {probability}")
    return birds_kind


def recognize_class(model, features):
    # features = np.expand_dims(features, 0)
    pred_proba, pred = make_prediction(model, features, BATCH_SIZE)
    # pred = make_prediction(model, features, 1)[0]
    return pred_proba[0], model.classes_dict[pred[0]]
