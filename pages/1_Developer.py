import streamlit as st
from PIL import Image
import os

st.set_page_config(page_title="О разработчике", page_icon="👤")
st.title("👤 О разработчике")
col1, col2 = st.columns([1, 2])

with col1:
    photo_path = "assets/developer_photo.jpg"
    image = Image.open(photo_path)
    st.image(image, caption="Разработчик", use_container_width=True)

with col2:
    st.markdown("""
    ### ФИО: **Кондяков Матвей Дмитриевич**
    
    ### Номер учебной группы: **ФИТ-241**
    
    ### Тема РГР: **«Разработка Web-приложения (дашборда) для инференса (вывода) моделей ML и анализа данных»**
    
    ---
    
    ### Контактная информация:
    - 📧 Email: matveik157@gmail.com
    - 🔗 GitHub: github.com/xxolodok
    """)
    