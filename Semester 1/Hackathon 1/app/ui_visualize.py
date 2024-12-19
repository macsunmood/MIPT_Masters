import streamlit as st


import os
# import ast
# import random
from PIL import Image
from io import BytesIO
from urllib.request import urlopen
# from dotenv import load_dotenv

import pandas as pd
import pydeck as pdk
import altair as alt
import matplotlib.pyplot as plt

from core import *


map_height = 500


# # Random color generation
# def random_color(alpha=160):
#     return [random.randint(0, 255) for _ in range(3)] + [alpha]

# Pick a predefined palette
PALETTE = plt.cm.tab20.colors
PALETTE = [tuple(int(c * 255) for c in color) + (160,) for color in PALETTE]  # convert to RGBA format


# Set colors based on sequence number
def color_from_palette(index, palette=PALETTE):
    return palette[index % len(palette)]


# Load data
st.session_state['file_selector_is_expanded'] = False
# if 'file_selector_is_expanded' not in st.session_state:
#     st.session_state['file_selector_is_expanded'] = False

file_selector_container = st.sidebar.expander(
    'Выбор файла', 
    expanded=False
    # expanded=st.session_state['file_selector_is_expanded']
)

global csvfile
# csvfile = None
csvfile = "./top_30.csv"

# Choose file upload mode
with file_selector_container:
    video_extensions = ['.csv']
    upload_mode = st.toggle('Local dir', help='Выбор между загрузкой и списком файлов локального каталога', value=True)

    if upload_mode:
        def file_selector(folder_path='.'):
            is_video_file = lambda f: any(f.lower().endswith(ext) for ext in video_extensions)

            video_files = [f for f in os.listdir(folder_path) if is_video_file(f)]

            if not video_files:
                st.warning('В выбранной директории не найдено CSV файлов.')
                return None

            selected_filename = st.selectbox('Выберите CSV файл', video_files, help=f'из {folder_path}')
            return os.path.join(folder_path, selected_filename)

        videofile = file_selector()
        # videofile_name = os.path.split(videofile)[-1]
        # file_path_input = st.text_input('CSV file path:', videofile)
    else:
        uploaded_video = st.file_uploader('Загрузить CSV файл', type=video_extensions)
        # videofile_name = uploaded_video.name if uploaded_video else ''
        if uploaded_video:
            csvfile = uploaded_video.name
            with open(videofile, mode='wb') as f:
                f.write(uploaded_video.read())  # save video to disk


data = load_data(csvfile)


# Check for the required columns
required_columns = {"latitude", "longitude", "common_name", "primary_label", "date"}
if not required_columns.issubset(data.columns):
    st.error(f"Файл {csvfile} должен содержать следующие столбцы: {', '.join(required_columns)}")
else:
    unique_species = data["common_name"].unique()

    # Assign colors to species in order
    # species_colors = {species: random_color() for species in unique_species}
    species_colors = {species: color_from_palette(i) for i, species in enumerate(unique_species)}
    data["color"] = data["common_name"].map(species_colors)

    # Main title
    # st.title("Интерактивная карта птиц")
    # st.write("На карте показано распределение птиц по широте и долготе.")
    # title_placeholder = st.title('')

    # Bird species selection widget
    species = st.sidebar.selectbox("Вид птицы 🦜", options=["Все"] + list(unique_species))
    specie_selected = species != "Все"

    # Filter by bird type
    if specie_selected:
        filtered_data = data[data["common_name"] == species]
        filtered_data = filtered_data[filtered_data["common_name"].str.strip().str.lower() == species.strip().lower()]
    else:
        filtered_data = data

    # Latitude and longitude selection widget inside an expandable container
    min_lat, max_lat = data["latitude"].min(), data["latitude"].max()
    min_lon, max_lon = data["longitude"].min(), data["longitude"].max()

    with st.sidebar.popover("Географический фильтр"):
        lat_range = st.slider("Диапазон широты", min_lat, max_lat, (min_lat, max_lat))
        lon_range = st.slider("Диапазон долготы", min_lon, max_lon, (min_lon, max_lon))

    filtered_data = filtered_data[
        (filtered_data["latitude"] >= lat_range[0]) & (filtered_data["latitude"] <= lat_range[1]) &
        (filtered_data["longitude"] >= lon_range[0]) & (filtered_data["longitude"] <= lon_range[1])
    ]

    # Date period picker widget
    min_date, max_date = pd.to_datetime(data['date']).min(), pd.to_datetime(data['date']).max()

    col_start_date, col_end_date = st.sidebar.columns([1, 1])
    start_date = col_start_date.date_input("Начало периода", min_date, min_value=min_date, max_value=max_date, format="DD.MM.YYYY")
    end_date = col_end_date.date_input("Конец периода", max_date, min_value=start_date, max_value=max_date, format="DD.MM.YYYY")
    
    filtered_data = filtered_data[
        (pd.to_datetime(filtered_data['date']) >= pd.to_datetime(start_date)) & 
        (pd.to_datetime(filtered_data['date']) <= pd.to_datetime(end_date))
    ]

    # Season selection widget
    selected_seasons = st.sidebar.multiselect(
        "Сезон",
        options=["Зима", "Весна", "Лето", "Осень"],
        default=["Весна", "Лето", "Осень", "Зима"],
        placeholder="Выберите сезоны"
    )
    
    # Dict for Russian-English season names
    season_translation = {
        "Зима": "Winter",
        "Весна": "Spring",
        "Лето": "Summer",
        "Осень": "Fall"
    }

    # Convert user-selected seasons to English
    selected_seasons_english = [season_translation[season] for season in selected_seasons]

    # Filter data by selected seasons
    if selected_seasons_english:
        filtered_data = filtered_data[filtered_data['season'].isin(selected_seasons_english)]

    # Filter by latitude and longitude
    filtered_data = filtered_data[
        (filtered_data["latitude"] >= lat_range[0]) & 
        (filtered_data["latitude"] <= lat_range[1]) &
        (filtered_data["longitude"] >= lon_range[0]) & 
        (filtered_data["longitude"] <= lon_range[1])
    ]

    # Convert 'date' column to datetime and filter by date
    filtered_data['date'] = pd.to_datetime(filtered_data['date'], errors='coerce')  # Преобразование с обработкой ошибок
    filtered_data = filtered_data.dropna(subset=['date'])  # Удаляем записи с некорректными датами
    filtered_data = filtered_data[
        (filtered_data['date'] >= pd.to_datetime(start_date)) & 
        (filtered_data['date'] <= pd.to_datetime(end_date))
    ]


    st.sidebar.success('✔️ CSV обработан успешно')  # ✔️❌❗⚠️✅


    # Pydeck visualization
    if not filtered_data.empty:
        view_state = pdk.ViewState(
            latitude=filtered_data["latitude"].mean(),
            longitude=filtered_data["longitude"].mean(),
            zoom=3,
            pitch=0
        )
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=filtered_data,
            get_position="[longitude, latitude]",
            get_color="color",
            get_radius=50000,
        )
        r = pdk.Deck(
            # map_style='mapbox://styles/mapbox/light-v6',
            layers=[layer],
            initial_view_state=view_state,
            tooltip={"text": "{common_name}"},
        )


        stats_data = {
            "Общее количество наблюдений": [len(filtered_data)],
            # "Количество уникальных видов птиц": [filtered_data["common_name"].nunique()],
            "Количество уникальных дней наблюдений": [filtered_data["date"].nunique()]
        }


        if specie_selected:
            col_map, col_info = st.columns([4, 2])
            col_stats, col_names = st.columns([4, 2])

            bird_image = get_bird_image(species)
            if bird_image:
                target_height = map_height - 110
                bird_image = Image.open(
                    BytesIO(urlopen(bird_image).read())
                    # requests.get(bird_image, stream=True).raw
                )
                img_width, img_height = bird_image.size
                img_fix_ratio = target_height / img_height
                col_info.image(bird_image, width=int(img_width * img_fix_ratio))#, use_container_width=True) caption=img_fix_ratio # species
            else:
                col_info.warning("Изображение не найдено.")

            col_names.subheader("")

            species_code = data[data["common_name"] == species]["primary_label"].iloc[0]
            bird_info = get_bird_info(species_code)

            if isinstance(bird_info, list) and bird_info:
                first_bird_info = bird_info[0]
                # st.sidebar.text('Названия птицы')
                # bird_names = f"**{species} ({first_bird_info.get('comName', 'Нет данных')})** **[`{first_bird_info.get('sciName', 'Нет данных')}`]**"
                # bird_names = f"{first_bird_info.get('comName', 'Нет данных')} | **`{first_bird_info.get('sciName', 'Нет данных')}`**"
                bird_names = f'''
                {first_bird_info.get('comName', '<Нет данных о имени>')} (**{species}**)

                **`{first_bird_info.get('sciName', '<Нет данных о научном названии>')}`**
                '''
                # "taxonOrder":28313
                # "order":"Passeriformes"
                # "familyCode":"turdid1"
                # "familyComName":"Thrushes and Allies"
                # "familySciName":"Turdidae"


                # title_placeholder.title(bird_names)
                # col_info.subheader(species)
                col_info.info(bird_names)
                # col_names.info(bird_names)
                # st.sidebar.info(bird_names)


                # bird_names = (f'''
                #     <div style='text-align: right; line-height: 1.5;'>
                #     <h>**{species}**<br></h>
                #     **{first_bird_info.get('comName', 'Нет данных')}**<br>
                #     **[{first_bird_info.get('sciName', 'Нет данных')}]**
                #     </div>"
                # ''')

                # col_info = st.container()  # Создание отдельного контейнера
                # col_info.markdown(bird_names, unsafe_allow_html=True)
            else:
                col_info.warning("Нет информации о птице.")

        else:
            col_map = st.container()
            col_stats = st.container()
            stats_data["Количество уникальных видов"] = [filtered_data["common_name"].nunique()]


        col_map.pydeck_chart(
            r,
            height=map_height,
            # width=700,
            # use_container_width=True
        )
        
        # Statistics
        col_stats.subheader("Статистика наблюдений 🐦")



        df_stats = pd.DataFrame(stats_data)
        col_stats.markdown(df_stats.style.hide(axis="index").to_html(), unsafe_allow_html=True)
        # col_stats.dataframe(df_stats, hide_index=True, use_container_width=True)

        # # Отступ
        # placeholder = st.empty()
        # placeholder.write("")


        # Проверка: выбран ли конкретный вид птицы
        if specie_selected:
            st.subheader("Динамика наблюдений и риск вымирания вида")

            # Getting the bird code
            bird_code = filtered_data[filtered_data["common_name"] == species]["primary_label"].iloc[0]

            # Call bird_dynamics
            df_bird_dynamics = bird_dynamics(
                df=data,
                bird=bird_code,
                longitude_left=lon_range[0],
                longitude_right=lon_range[1],
                latitude_min=lat_range[0],
                latitude_max=lat_range[1],
                start_date=start_date,
                end_date=end_date,
                selected_seasons=selected_seasons_english,
            )

            if df_bird_dynamics is not None:
                # Styling
                def color_rows(row):
                    """Returns styles for a string based on risk level."""
                    color = {
                        'Нет данных': 'background-color: lightgray;',
                        'Низкий': 'background-color: lightgreen;',
                        'Средний': 'background-color: yellow;',
                        'Высокий': 'background-color: lightcoral;'
                    }.get(row['Риск вымирания'], '')
                    return [color] * len(row)

                # Apply the style and show dataframe
                st.dataframe(df_bird_dynamics.style.apply(color_rows, axis=1), 
                             use_container_width=True)


                # Create a field to display the risk as a number
                risk_map = {'Нет данных': 0, 'Низкий': 1, 'Средний': 2, 'Высокий': 3}
                df_bird_dynamics['Риск_число'] = df_bird_dynamics['Риск вымирания'].map(risk_map)
                # Create a heat map
                heatmap = alt.Chart(df_bird_dynamics).mark_rect().encode(
                    x=alt.X('Год:O', title='Год'),
                    y=alt.Y('Риск_число:N', title='Риск вымирания', sort='descending'),

                    color=alt.Color(
                        'Риск вымирания:N',
                        scale=alt.Scale(
                            domain=['Нет данных', 'Низкий', 'Средний', 'Высокий'][::-1],
                            range=['gray', 'green', 'orange', 'red'][::-1],
                        ),
                        title='Риск вымирания'
                    ),

                    tooltip=['Год', 'Частота', 'Количество записей вида', 'Риск вымирания']
                ).properties(
                    # title='Тепловая карта наблюдений и риска вымирания',
                    height=300
                )
                st.altair_chart(heatmap, use_container_width=True)

                chart = (
                    alt.Chart(df_bird_dynamics)
                    .mark_area(opacity=0.3)
                    .encode(
                        x="Год",
                        # x=alt.X("Год:T"),
                        y=alt.Y("Частота:Q", title='Частота наблюдений', stack=None),
                        color=alt.Color('Риск вымирания:N', scale=alt.Scale(
                            domain=['Нет данных', 'Низкий', 'Средний', 'Высокий'][::-1],
                            range=['gray', 'green', 'orange', 'red'][::-1]
                        ), title='Риск вымирания')
                    )
                )
                st.altair_chart(chart, use_container_width=True)

            else:
                st.warning("Недостаточно данных для анализа выбранного вида птицы.")
    else:
        st.warning("Нет данных для отображения на карте.")
