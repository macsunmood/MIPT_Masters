import streamlit as st

import os
import re
import zipfile
from pathlib import Path
import gdown

import numpy as np
import pandas as pd

import requests
from dotenv import load_dotenv

import tensorflow as tf
from keras.models import load_model
# from keras_cv.models import ImageClassifier

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

import librosa


class CFG:
    seed = 42
    
    # Input image size and batch size
    img_size = [128, 384]
    batch_size = 64
    
    # Audio duration, sample rate, and length
    duration = 15  # seconds
    sample_rate = 32000
    audio_len = duration * sample_rate
    
    # STFT parameters
    nfft = 2028
    window = 2048
    hop_length = audio_len // (img_size[1] - 1)
    fmin = 20
    fmax = 16000
    
    # Number of epochs, model name
    epochs = 10
    preset = 'efficientnetv2_b2_imagenet'
    
    # Data augmentation parameters
    augment = True


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# from utils import *


# Load environment variables from .env file
load_dotenv()

# Access the API credentials
EBIRD_API_URL = os.getenv("EBIRD_API_URL")
EBIRD_API_KEY = os.getenv("EBIRD_API_KEY")
WIKIMEDIA_API_URL = os.getenv("WIKIMEDIA_API_URL")

df_perch_labels = pd.read_csv('perch_labels.csv')

from settings import MODELS_GDRIVE


# Load data from CSV
@st.cache_data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)

        # if "type" in data.columns:
        #     data["type"] = data["type"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        # data['primary_label'] = data['primary_label'].astype(str)

        return data
    except FileNotFoundError:
        st.error(f"Файл {file_path} не найден. Проверьте путь и повторите попытку.")
        return pd.DataFrame()


# Retrieve bird info via eBird API
@st.cache_data
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
@st.cache_data
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
def bird_dynamics(df, bird='', 
                  longitude_left=-180, longitude_right=180, 
                  latitude_min=-90, latitude_max=90, 
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


###==============================================
###
### ML PART
###


CLASSES_DICT = {
    0: 'gnwtea', 
    1: 'grasal1', 
    2: 'grnjay', 
    3: 'reevir1', 
    4: 'heptan'
}

MODELS_DIR = 'models'
MODELS_LIST = ['perch_v4', 'efficientnetv2_b2']


# def get_classes_dict(model):
#     d = next(l.classes_dict for l in model.layers 
#              if hasattr(l, 'classes_dict'))
#     return {int(k): v for k, v in d.items()}


@st.cache_resource
def load_models(models_urls=MODELS_GDRIVE, models_dir=MODELS_DIR):
    '''Load a list of pretrained models'''
    # Open a new TensorFlow session
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    session = tf.compat.v1.Session(config=config)

    for dir_ in MODELS_LIST:
        if not os.path.exists(dir_):
            os.makedirs(dir_, exist_ok=True)

    with session.as_default():
        models = {}
        save_dest = Path(models_dir)
        save_dest.mkdir(exist_ok=True)

        if models_urls:
            models_list = download_models(models_urls=models_urls, models_dir=models_dir)
        else:
            # Load from the local dir
            models_list = os.listdir(models_dir)

        for m_name in models_list:
            if 'perch' in m_name:
                # model_url = 'https://kaggle.com/models/google/bird-vocalization-classifier/frameworks/tensorFlow2/variations/bird-vocalization-classifier/versions/4'
                # model = hub.load(model_url)
                model = tf.saved_model.load(f'{MODELS_DIR}\{m_name}')
            if 'efficientnet' in m_name:
                d = f'{MODELS_DIR}\{m_name}'
                model = load_model(os.path.join(d, [f for f in os.listdir(d) if f.endswith(".keras")][0]))
            if model:
                models.update({m_name: model})
    return session, models


def download_models(models_urls=MODELS_GDRIVE, models_dir=MODELS_DIR):
    '''Download and extract pretrained models'''
    os.makedirs(models_dir, exist_ok=True)
    models_list = []

    # Download each model
    for model_name, model_url in models_urls.items():
        # zip_path = os.path.join(models_dir, f"{model_name}.zip")  # path to .zip-files
        m_path = os.path.join(MODELS_DIR, model_name)
        if not os.path.exists(m_path):
            os.makedirs(m_path, exist_ok=True)
            with st.spinner(f"Downloading model {model_name}... this may take a while! \n Don't stop!"):
                # Download the file from Google Drive
                gdown.download(model_url, f'{models_dir}/{model_name}', quiet=False)
        # # Extract the .zip file
        # if zipfile.is_zipfile(models_dir):
        #     with st.spinner(f"Extracting model {model_name}..."):
        #         with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        #             zip_ref.extractall(models_dir)
        models_list.append(model_name)
    return models_list


def download_xeno_canto_audio(folder_path, filename):
    # Record ID - search for a number in the file name
    match = re.search(r'\d+', filename)
    if match:
        record_id = match.group()

    # URL API
    api_url = f"https://www.xeno-canto.org/api/2/recordings?query=nr:{record_id}"

    # API request
    response = requests.get(api_url)
    data = response.json()

    # Get the file link
    if data['recordings']:
        file_url = data['recordings'][0]['file']
        print(f"Ссылка на аудиофайл: {file_url}")

        # Download an audio file
        audio_response = requests.get(file_url)
        new_file_name = f"XC{record_id}.ogg"
        with open(os.path.join(folder_path, new_file_name), "wb") as f:
            f.write(audio_response.content)
        print(f"Файл {new_file_name}.mp3 сохранен.")
    else:
        print("Запись не найдена.")


# Decodes Audio
def build_decoder(with_labels=True, dim=1024):
    def get_audio(filepath):
        audio, sr = librosa.load(filepath, sr=CFG.sample_rate)
        audio = tf.cast(audio, tf.float32)

        # Check if audio is 1D (mono) or 2D (stereo)
        audio_shape = tf.shape(audio)
        if len(audio_shape) > 1 and audio_shape[1] > 1:  # stereo -> mono
            audio = audio[..., 0:1]

        return audio

    def crop_or_pad(audio, target_len, pad_mode="constant"):
        audio_len = tf.shape(audio)[0]
        diff_len = abs(
            target_len - audio_len
        )  # find difference between target and audio length
        if audio_len < target_len:  # do padding if audio length is shorter
            pad1 = tf.random.uniform([], maxval=diff_len, dtype=tf.int32)
            pad2 = diff_len - pad1
            audio = tf.pad(audio, paddings=[[pad1, pad2]], mode=pad_mode)
        elif audio_len > target_len:  # do cropping if audio length is larger
            idx = tf.random.uniform([], maxval=diff_len, dtype=tf.int32)
            audio = audio[idx : (idx + target_len)]
        return tf.reshape(audio, [target_len])

    def apply_preproc(spec):
        # Standardize
        mean = tf.math.reduce_mean(spec)
        std = tf.math.reduce_std(spec)
        spec = tf.where(tf.math.equal(std, 0), spec - mean, (spec - mean) / std)

        # Normalize using Min-Max
        min_val = tf.math.reduce_min(spec)
        max_val = tf.math.reduce_max(spec)
        spec = tf.where(
            tf.math.equal(max_val - min_val, 0),
            spec - min_val,
            (spec - min_val) / (max_val - min_val),
        )
        return spec

    def get_target(target):
        target = tf.reshape(target, [1])
        target = tf.cast(tf.one_hot(target, CFG.num_classes), tf.float32)
        target = tf.reshape(target, [CFG.num_classes])
        return target

    def decode(path):
        # Load audio file
        audio = get_audio(path)
        # Crop or pad audio to keep a fixed length
        audio = crop_or_pad(audio, dim)
        # Audio to Spectrogram
        spec = tf.keras.layers.MelSpectrogram(
            num_mel_bins=CFG.img_size[0],
            fft_length=CFG.nfft,
            sequence_stride=CFG.hop_length,
            sampling_rate=CFG.sample_rate,
        )(audio)
        # Apply normalization and standardization
        spec = apply_preproc(spec)
        # Spectrogram to 3 channel image (for imagenet)
        spec = tf.tile(spec[..., None], [1, 1, 3])
        spec = tf.reshape(spec, [*CFG.img_size, 3])
        return spec

    def decode_with_labels(path, label):
        label = get_target(label)
        return decode(path), label

    return decode_with_labels if with_labels else decode


def audio_to_melspectrogram(audio, sr, n_mels=256):
    """
    Преобразует аудиосигнал в мел-спектрограмму.
    """
    spec = librosa.feature.melspectrogram(
        y=audio, 
        sr=sr, 
        n_mels=n_mels, 
        n_fft=2048,
        fmax=sr // 2,
        hop_length=512,
    )

    log_spec = librosa.power_to_db(spec, ref=1.0)  # перевод в dB
    min_ = log_spec.min()
    max_ = log_spec.max()
    if max_ != min_:
        log_spec = (log_spec - min_) / (max_ - min_)
    return log_spec


def predict_species(csv_name, filename, model, option_model):
    def resample_rate(audio, sample_rate, new_sample_rate=32000):
        """Изменяет sample rate аудио на 32000 Hz, если это необходимо."""
        if sample_rate != new_sample_rate:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=new_sample_rate)
        return audio, new_sample_rate

    def break_into_frames(audio, sample_rate, window_size=5.0, step_size=5.0):
        """Разбивает аудио на фреймы по 5 секунд (модель Perch обучена на работе с 5-сек. записями)"""
        frame_length = int(window_size * sample_rate)
        frame_step = int(step_size * sample_rate)
        framed_audio = tf.signal.frame(audio, frame_length, frame_step, pad_end=True)
        return framed_audio

    name, ext = os.path.splitext(csv_name)
    folder_name = f"{name}__{ext.lstrip('.')}"
    folder_path = os.path.join("audio", folder_name)
    full_path = os.path.join(folder_path, filename)

    # Download audio file if not exists
    if not os.path.exists(full_path):
        print(f'Файл {full_path} не найден! Попробуем скачать..')
        os.makedirs(folder_path, exist_ok=True)
        download_xeno_canto_audio(folder_path, filename)

    # Загружаем аудио и преобразуем в фреймы
    audio, sr = librosa.load(full_path)
    audio, sr = resample_rate(audio, sr)

    if 'perch' in option_model:
        framed_audio = break_into_frames(audio, sr)  # разбиваем на фреймы
        logits, embeddings = model.infer_tf(framed_audio[0:1])  # получаем прогноз для первого фрейма
        probs = tf.nn.softmax(logits)

    elif 'efficientnet' in option_model:
        # Предсказание для первого фрейма
        # framed_audio = break_into_frames(audio, sr, window_size=15.0, step_size=15.0)  # у EfficientNet 15 сек)
        # first_frame = framed_audio[0].numpy()  # преобразуем первый фрейм в мел-спектрограмму

        decoder = build_decoder(with_labels=False, dim=CFG.audio_len)
        # Декодируем аудиофайл в спектрограмму
        spec = decoder(full_path)
        # Убедимся, что спектрограмма имеет правильный формат
        if tf.is_tensor(spec):
            spec = spec.numpy()  # Преобразуем в NumPy, если это TensorFlow тензор
        # Добавляем размерность батча
        spec = np.expand_dims(spec, axis=0)

        # Используем Keras для предсказания
        logits = model.predict(spec)
        
        # Преобразуем выход в вероятности
        probs = tf.nn.softmax(logits).numpy()  # применяем softmax для получения вероятностей

    # Получаем предсказанный класс
    argmax = np.argmax(probs)

    if 'perch' in option_model:
        birds_kind = df_perch_labels.iloc[argmax]['ebird2021']
    else:
        birds_kind = CLASSES_DICT[argmax]

    probability = probs[0][argmax]

    return birds_kind, probability


class MockResource:
    RLIMIT_AS = 0
    RLIMIT_DATA = 0
    RLIMIT_FSIZE = 0
    RLIMIT_NOFILE = 0

    @staticmethod
    def getrlimit(limit):
        return (0, 0)


resource = MockResource()
