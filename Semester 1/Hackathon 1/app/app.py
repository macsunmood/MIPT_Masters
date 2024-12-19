import streamlit as st


# Page config
st.set_page_config(
    # page_title="Интерактивная карта птиц", 
    page_icon=':parrot:', 
    layout="wide",
    menu_items={
        'Get help': 'https://www.kaggle.com/c/birdclef-2021', 
        'About': "# MIPT Master's :: Hackathon 1. Team 3"
    },
    # initial_sidebar_state="collapsed"
)
# st.logo('mipt_logo.png', size='large')
# st.logo('mipt_logo_wide.png', size='large', icon_image='mipt_logo.png', link='https://mipt.online/masters/data_science')


ui_classify_page = st.Page("ui_classify.py", title="Распознать", icon="🔍")#":material/delete:")
ui_visualize_page = st.Page("ui_visualize.py", title="Визуализировать", icon="✨")#":material/add_circle:")

pg = st.navigation([ui_classify_page, ui_visualize_page])
pg.run()
