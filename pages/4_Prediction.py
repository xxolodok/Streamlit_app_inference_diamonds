import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io

st.set_page_config(page_title="Предсказание цены", page_icon="💎")
st.title("💎 Предсказание цены бриллианта")

MODEL_FILES = {
    "Bagging Regressor": "BaggingRegressor.joblib",
    "CatBoost Regressor": "CatBoost.joblib",
    "Gradient Boosting Regressor": "GradientBoostingRegressor.joblib",
    "Polynomial Regression (ElasticNet)": "PolyRegression.joblib",
    "Stacking Regressor": "StackingRegressor.joblib"
}

FEATURE_NAMES = ['carat', 'cut', 'color', 'clarity', 'depth', 'table']

@st.cache_resource
def load_all_models():
    models = {}
    try:
        for name, filename in MODEL_FILES.items():
            try:
                models[name] = joblib.load(f'models/{filename}')
            except Exception as e:
                st.warning(f"Не удалось загрузить модель {name}: {str(e)}")
                models[name] = None
        return models
    except Exception as e:
        st.error(f"❌ Ошибка загрузки: {str(e)}")
        return None

models_dict = load_all_models()

available_models = []
if models_dict is not None:
    available_models = [name for name, model in models_dict.items() if model is not None]

# Словари для преобразования категорий
cut_mapping = {'Ideal': 1, 'Premium': 2, 'Very Good': 3, 'Good': 4, 'Fair': 5}
color_mapping = {'D': 1, 'E': 2, 'F': 3, 'G': 4, 'H': 5, 'I': 6, 'J': 7}
clarity_mapping = {'IF': 1, 'VVS1': 2, 'VVS2': 3, 'VS1': 4, 'VS2': 5, 'SI1': 6, 'SI2': 7, 'I1': 8}

st.markdown("---")
st.subheader("Выбор модели машинного обучения")

if available_models:
    col_model1, col_model2 = st.columns([2, 1])
    with col_model1:
        selected_model_name = st.selectbox(
            "Выберите модель для предсказания цены:",
            options=available_models,
        )
        selected_model = models_dict[selected_model_name]
    with col_model2:
        model_info = {
            "Bagging Regressor": "📊 Ансамбль из деревьев",
            "CatBoost Regressor": "🐱 Градиентный бустинг",
            "Gradient Boosting Regressor": "📈 Градиентный бустинг",
            "Polynomial Regression (ElasticNet)": "📐 Полиномиальная регрессия (лучшая)",
            "Stacking Regressor": "🏗️ Комбинация моделей"
        }
        st.info(model_info.get(selected_model_name, "Выберите модель"))
else:
    selected_model = None
    st.error("❌ Не удалось загрузить ни одну модель")
    st.stop()

st.markdown("---")
st.markdown("### Введите характеристики бриллианта")

tab1, tab2 = st.tabs(["📝 Ручной ввод", "📁 Загрузка CSV"])

with tab1:
    st.subheader("Ввод характеристик бриллианта")
    
    col1, col2 = st.columns(2)
    
    with col1:
        carat = st.number_input("Вес (carat)", min_value=0.2, max_value=5.0, value=1.0, step=0.1)
        cut = st.selectbox("Огранка (cut)", options=list(cut_mapping.keys()))
        color = st.selectbox("Цвет (color)", options=list(color_mapping.keys()))
    
    with col2:
        clarity = st.selectbox("Чистота (clarity)", options=list(clarity_mapping.keys()))
        depth = st.number_input("Глубина (depth), %", min_value=50.0, max_value=75.0, value=61.0, step=0.5)
        table = st.number_input("Площадка (table), %", min_value=43.0, max_value=80.0, value=57.0, step=0.5)
    
    if st.button("🔮 Предсказать цену", type="primary", key="manual_predict"):
        if selected_model is not None:
            log_carat = np.log(carat)
            
            input_data = pd.DataFrame([{
                'carat': log_carat,
                'cut': cut_mapping[cut],
                'color': color_mapping[color],
                'clarity': clarity_mapping[clarity],
                'depth': depth,
                'table': table
            }])
            
            prediction_log = selected_model.predict(input_data)[0]
            
            prediction_price = np.exp(prediction_log)
            
            st.success(f"""
            ### 💎 Результат предсказания:
            **Модель:** {selected_model_name}
            
            #### **${prediction_price:,.2f} USD**
            """)
            
            with st.expander("📊 Дополнительная информация"):
                st.markdown(f"""
                - **Вес:** {carat} карат (log: {log_carat:.4f})
                - **Огранка:** {cut} (код: {cut_mapping[cut]})
                - **Цвет:** {color} (код: {color_mapping[color]})
                - **Чистота:** {clarity} (код: {clarity_mapping[clarity]})
                - **Глубина:** {depth}%
                - **Площадка:** {table}%
                - **Предсказанный log(price):** {prediction_log:.4f}
                """)
        else:
            st.error("Модель не загружена")

with tab2:
    st.subheader("Загрузка CSV-файла с данными")
    
    st.markdown("""
    **Требования к CSV-файлу:**
    - Файл должен содержать столбцы: `carat`, `cut`, `color`, `clarity`, `depth`, `table`
    - Категориальные значения: Ideal, Premium, Very Good, Good, Fair; D, E, F, G, H, I, J; IF, VVS1, VVS2, VS1, VS2, SI1, SI2, I1
    - Разделитель: запятая
    """)
    
    sample_df = pd.DataFrame({
        'carat': [0.75, 1.2, 0.5],
        'cut': ['Ideal', 'Premium', 'Very Good'],
        'color': ['D', 'G', 'F'],
        'clarity': ['VS1', 'SI1', 'VVS2'],
        'depth': [61.5, 62.0, 60.8],
        'table': [56.0, 58.0, 55.0]
    })
    
    csv_buffer = io.StringIO()
    sample_df.to_csv(csv_buffer, index=False)
    
    st.download_button(
        label="📥 Скачать пример CSV-файла",
        data=csv_buffer.getvalue(),
        file_name="diamonds_sample.csv",
        mime="text/csv"
    )
    
    uploaded_file = st.file_uploader("Выберите CSV-файл", type=['csv'])
    
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Загружено {len(uploaded_df)} записей")
            st.dataframe(uploaded_df.head(), use_container_width=True)
            
            required_columns = ['carat', 'cut', 'color', 'clarity', 'depth', 'table']
            missing_columns = [col for col in required_columns if col not in uploaded_df.columns]
            
            if missing_columns:
                st.error(f"❌ Отсутствуют столбцы: {missing_columns}")
            else:
                processed_df = uploaded_df.copy()
                
                processed_df['cut'] = processed_df['cut'].map(cut_mapping)
                processed_df['color'] = processed_df['color'].map(color_mapping)
                processed_df['clarity'] = processed_df['clarity'].map(clarity_mapping)
                processed_df['carat'] = np.log(processed_df['carat'])
                
                if processed_df[['cut', 'color', 'clarity']].isnull().any().any():
                    st.error("❌ Некорректные значения в категориальных столбцах")
                else:
                    if selected_model is not None:
                        predictions_log = selected_model.predict(processed_df[FEATURE_NAMES])
                        predictions_price = np.exp(predictions_log)
                        
                        display_df = uploaded_df.copy()
                        display_df['predicted_price'] = predictions_price
                        display_df['predicted_price'] = display_df['predicted_price'].apply(lambda x: f"${x:,.2f}")
                        
                        st.subheader("📊 Результаты предсказаний")
                        st.dataframe(display_df, use_container_width=True)
                        
                        output_df = display_df.copy()
                        output_csv = output_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Скачать результаты",
                            data=output_csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
                        
                        st.subheader("📈 Статистика предсказаний")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Минимальная цена", f"${predictions_price.min():,.2f}")
                        with col2:
                            st.metric("Средняя цена", f"${predictions_price.mean():,.2f}")
                        with col3:
                            st.metric("Максимальная цена", f"${predictions_price.max():,.2f}")
                    else:
                        st.error("Модель не загружена")
        except Exception as e:
            st.error(f"Ошибка: {str(e)}")
            