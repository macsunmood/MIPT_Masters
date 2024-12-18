import os
import ast
import random
import requests
from dotenv import load_dotenv

import streamlit as st
import pandas as pd
import pydeck as pdk
import matplotlib.pyplot as plt

from core import *


# Page config
st.set_page_config(
    # page_title="Интерактивная карта птиц", 
    page_icon=':parrot:', 
    layout="wide",
    menu_items={
        'Get help': 'https://www.kaggle.com/c/birdclef-2021', 
        'About': "# MIPT Master's :: Hackathon 1. Team 3"
    }
)
# st.logo('mipt_logo.png', size='large')
st.logo('mipt_logo_wide.png', size='large', icon_image='mipt_logo.png', link='https://mipt.online/masters/data_science')


# # Random color generation
# def random_color(alpha=160):
#     return [random.randint(0, 255) for _ in range(3)] + [alpha]

# Pick a predefined palette
PALETTE = plt.cm.tab20.colors
PALETTE = [tuple(int(c * 255) for c in color) + (160,) for color in PALETTE]  # convert to RGBA format


# Set colors based on sequence number
def color_from_palette(index, palette=PALETTE):
    return palette[index % len(palette)]


def main():
    # Load data
    if 'file_selector_is_expanded' not in st.session_state:
        st.session_state['file_selector_is_expanded'] = True

    file_selector_container = st.sidebar.expander(
        'Выбрать файл', 
        expanded=st.session_state['file_selector_is_expanded']
    )

    global csvfile
    csvfile = None

    # Choose file upload mode
    with file_selector_container:
        video_extensions = ['.csv']
        upload_mode = st.toggle('Local dir', help='Choosing between uploading and local directory files list', value=True)

        if upload_mode:
            def file_selector(folder_path='.'):
                is_video_file = lambda f: any(f.lower().endswith(ext) for ext in video_extensions)

                video_files = [f for f in os.listdir(folder_path) if is_video_file(f)]

                if not video_files:
                    st.warning('No video files found in the selected directory.')
                    return None

                selected_filename = st.selectbox('Select a CSV file', video_files, help=f'from {folder_path}')
                return os.path.join(folder_path, selected_filename)

            videofile = file_selector()
            # videofile_name = os.path.split(videofile)[-1]
            # file_path_input = st.text_input('CSV file path:', videofile)
        else:
            uploaded_video = st.file_uploader('Upload a CSV', type=video_extensions)
            # videofile_name = uploaded_video.name if uploaded_video else ''
            if uploaded_video:
                videofile = uploaded_video.name
                with open(videofile, mode='wb') as f:
                    f.write(uploaded_video.read())  # save video to disk

    csvfile = "./top_30.csv"
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
        filtered_data = data if species == "Все" else data[data["common_name"] == species]
        specie_selected = species != "Все"

        # Latitude and longitude selection widget
        min_lat, max_lat = data["latitude"].min(), data["latitude"].max()
        min_lon, max_lon = data["longitude"].min(), data["longitude"].max()
        lat_range = st.sidebar.slider("Диапазон широты", min_lat, max_lat, (min_lat, max_lat))
        lon_range = st.sidebar.slider("Диапазон долготы", min_lon, max_lon, (min_lon, max_lon))
        filtered_data = filtered_data[(filtered_data["latitude"] >= lat_range[0]) & (filtered_data["latitude"] <= lat_range[1]) &
                                       (filtered_data["longitude"] >= lon_range[0]) & (filtered_data["longitude"] <= lon_range[1])]

        # Date picker widget
        min_date, max_date = pd.to_datetime(data['date']).min(), pd.to_datetime(data['date']).max()

        col_start_date, col_end_date = st.sidebar.columns([1, 1])
        start_date = col_start_date.date_input("Начало периода", min_date, min_value=min_date, max_value=max_date, format="DD.MM.YYYY")
        end_date = col_end_date.date_input("Конец периода", max_date, min_value=start_date, max_value=max_date, format="DD.MM.YYYY")
        filtered_data = filtered_data[(pd.to_datetime(filtered_data['date']) >= pd.to_datetime(start_date)) & 
                                       (pd.to_datetime(filtered_data['date']) <= pd.to_datetime(end_date))]

        # Season selection widget
        selected_seasons = st.sidebar.multiselect(
            "Сезон",
            options=["Зима", "Весна", "Лето", "Осень"],
            default=["Весна", "Лето", "Осень", "Зима"],
            placeholder="Выберите сезоны"
        )
        
         # Словарь соответствий между русскими и английскими названиями сезонов
        season_translation = {
            "Зима": "Winter",
            "Весна": "Spring",
            "Лето": "Summer",
            "Осень": "Fall"
        }

        # Фильтрация по виду птицы
        if specie_selected:
            filtered_data = filtered_data[
                filtered_data["common_name"].str.strip().str.lower() == species.strip().lower()
            ]

        # Преобразуем выбранные пользователем сезоны в английские
        selected_seasons_english = [season_translation[season] for season in selected_seasons]

        # Фильтрация данных по выбранным сезонам
        if selected_seasons_english:
            filtered_data = filtered_data[filtered_data['season'].isin(selected_seasons_english)]

        # Фильтрация по широте и долготе
        filtered_data = filtered_data[
            (filtered_data["latitude"] >= lat_range[0]) & 
            (filtered_data["latitude"] <= lat_range[1]) &
            (filtered_data["longitude"] >= lon_range[0]) & 
            (filtered_data["longitude"] <= lon_range[1])
        ]

        # Преобразование столбца 'date' в datetime и фильтрация по дате
        filtered_data['date'] = pd.to_datetime(filtered_data['date'], errors='coerce')  # Преобразование с обработкой ошибок
        filtered_data = filtered_data.dropna(subset=['date'])  # Удаляем записи с некорректными датами
        filtered_data = filtered_data[
            (filtered_data['date'] >= pd.to_datetime(start_date)) & 
            (filtered_data['date'] <= pd.to_datetime(end_date))
        ]


        st.sidebar.success('CSV обработан успешно')  # ✔️⚠️❗✅  ✅     ❌


        # Pydeck визуализация
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
                layers=[layer],
                initial_view_state=view_state,
                tooltip={"text": "{common_name}"},
            )


            if specie_selected:
                col_map, col_info = st.columns([4, 2])
                col_map.pydeck_chart(r)

                bird_image = get_bird_image(species)
                if bird_image:
                    col_info.image(bird_image, caption=species)
                else:
                    col_info.warning("Изображение не найдено.")

                col_info.subheader("Названия")

                species_code = data[data["common_name"] == species]["primary_label"].iloc[0]
                bird_info = get_bird_info(species_code)

                if isinstance(bird_info, list) and bird_info:
                    first_bird_info = bird_info[0]
                    # st.sidebar.text('Названия птицы')
                    # bird_names = f"**{species} ({first_bird_info.get('comName', 'Нет данных')})** **[`{first_bird_info.get('sciName', 'Нет данных')}`]**"
                    bird_names = f"{first_bird_info.get('comName', 'Нет данных')} | **`{first_bird_info.get('sciName', 'Нет данных')}`**"
                    # title_placeholder.title(bird_names)
                    # col_info.subheader(species)
                    col_info.info(bird_names)


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
                st.pydeck_chart(r)


            
            # Статистика
            col_map.subheader("Статистика наблюдений 🐦")
            total_observations = len(filtered_data)
            unique_species_count = filtered_data["common_name"].nunique()
            observation_dates = filtered_data["date"].nunique()

            stats_data = {
                "Общее количество наблюдений": [total_observations],
                "Количество уникальных видов птиц": [unique_species_count],
                "Количество уникальных дней наблюдений": [observation_dates]
            }

            df_stats = pd.DataFrame(stats_data)
            col_map.markdown(df_stats.style.hide(axis="index").to_html(), unsafe_allow_html=True)
            # st.table(df_stats.style.hide(axis="index"), )
            # st.dataframe(df_stats, hide_index=True, use_container_width=False)

            # Отступ
            placeholder = st.empty()
            placeholder.write("")

            # Проверка: выбран ли конкретный вид птицы
            if specie_selected:
                st.subheader(f"Динамика наблюдений и риск вымирания вида")

                # Получение кода птицы
                bird_code = filtered_data[filtered_data["common_name"] == species]["primary_label"].iloc[0]

                # Вызов bird_dynamics
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
                    # st.write(df_bird_dynamics)
                    st.dataframe(df_bird_dynamics, use_container_width=True)

                    import altair as alt
                    chart = (
                        alt.Chart(df_bird_dynamics)
                        .mark_area(opacity=0.3)
                        .encode(
                            x="Год",
                            y=alt.Y("Risk:Q", stack=None),
                            # color="Риск вымирания:",
                        )
                    )
                    st.altair_chart(chart, use_container_width=True)

                else:
                    st.warning("Недостаточно данных для анализа выбранного вида птицы.")
        else:
            st.warning("Нет данных для отображения на карте.")

        # # Отображаем информацию о выбранной птице
        # if specie_selected:
        #     # st.sidebar.subheader(f"Информация о птице: {species}")
        #     # st.sidebar.subheader('Информация о птице')
        #     st.sidebar.divider()
            
        #     species_code = data[data["common_name"] == species]["primary_label"].iloc[0]
        #     bird_info = get_bird_info(species_code)

            # if isinstance(bird_info, list) and bird_info:
            #     first_bird_info = bird_info[0]
            #     # st.sidebar.text('Названия птицы')
            #     st.sidebar.write(f"**{first_bird_info.get('comName', 'Нет данных')}**")
            #     st.sidebar.write(f"[`{first_bird_info.get('sciName', 'Нет данных')}`]")

            #     # st.sidebar.write(f"**Научное название:** **`{first_bird_info.get('sciName', 'Нет данных')}`**")
            #     # st.sidebar.write(f"**Обиходное название:** **`{first_bird_info.get('comName', 'Нет данных')}`**")
                
            #     # df_info = pd.DataFrame.from_dict(
            #     #     {
            #     #         "Научное": "**`{first_bird_info.get('sciName', 'Нет данных')}`**",
            #     #         "Обиходное": "**`{first_bird_info.get('comName', 'Нет данных')}`**"
            #     #     }, 
            #     #     orient='index', 
            #     # )

            #     # st.sidebar.dataframe(df_info, width=500)
            #     # st.sidebar.markdown(df_info.style.hide(axis="columns").to_html(), unsafe_allow_html=True)
            # else:
            #     st.write("Нет информации о птице.")

            # bird_image = get_bird_image(species)
            # if bird_image:
            #     st.sidebar.image(bird_image, caption=species)
            # else:
            #     st.sidebar.write("Изображение не найдено.")


if __name__ == "__main__":
    main()
