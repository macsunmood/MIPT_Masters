import streamlit as st


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

pg = st.navigation([st.Page("ui_visualize.py"), st.Page("ui_classify.py")])
pg.run()
