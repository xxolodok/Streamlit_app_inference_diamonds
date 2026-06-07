import streamlit as st

st.title("💎 Diamond Price Predictor")
st.markdown("---")
st.markdown("""
### Добро пожаловать в дашборд предсказания цены бриллиантов!

Этот дашборд разработан в рамках выполнения расчетно-графической работы 
по дисциплине «Машинное обучение».

---
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("""
    ### **Анализ данных**
    
    - Визуализация зависимостей
    - Статистический анализ
    - Корреляционный анализ
    """)

with col2:
    st.success("""
    ### **ML модели**
    
    - Random Forest
    - Gradient Boosting
    - CatBoost
    - Stacking Ensemble
    """)

with col3:
    st.warning("""
    ### **Предсказание**
    
    - Ручной ввод данных
    - Загрузка CSV файлов
    - Мгновенный результат
    """)

st.markdown("---")
