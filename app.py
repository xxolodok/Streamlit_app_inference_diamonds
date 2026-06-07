
import streamlit as st

st.set_page_config(
    page_title="Diamond Price Predictor",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

home_page = st.Page(
    "home.py",
    title="Главная",
    default=True 
)

about_page = st.Page(
    "pages/1_Developer.py",
    title="О разработчике"
)

dataset_page = st.Page(
    "pages/2_Dataset.py",
    title="О наборе данных",
)

visuals_page = st.Page(
    "pages/3_Visualizations.py",
    title="Визуализации",
)

prediction_page = st.Page(
    "pages/4_Prediction.py",
    title="Предсказание цены"
)

pg = st.navigation([
    home_page, 
    about_page, 
    dataset_page, 
    visuals_page, 
    prediction_page
])

pg.run()
