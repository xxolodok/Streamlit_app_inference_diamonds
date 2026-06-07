import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Визуализации", page_icon="📈")
st.title("📈 Визуализации зависимостей в наборе данных")

@st.cache_data
def load_data():
    df = pd.read_csv('../data/diamonds_after_eda.csv')
    return df

try:
    df = load_data()
    data_loaded = True
except FileNotFoundError:
    st.warning("Файл данных не найден. Использую демонстрационные данные.")
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'carat': np.random.exponential(0.5, n) + 0.2,
        'price': np.random.exponential(2000, n) + 500,
        'cut': np.random.choice(['Ideal', 'Premium', 'Very Good', 'Good', 'Fair'], n),
        'color': np.random.choice(['D', 'E', 'F', 'G', 'H', 'I', 'J'], n),
        'clarity': np.random.choice(['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'], n),
        'depth': np.random.uniform(55, 70, n),
        'table': np.random.uniform(50, 70, n)
    })
    data_loaded = False

# 1. Scatter plot: carat vs price с цветом по cut
st.subheader("1. Зависимость цены от веса (carat)")

fig1 = px.scatter(
    df.sample(min(2000, len(df))), 
    x='carat', 
    y='price', 
    color='cut',
    title='Цена vs Вес (по категориям огранки)',
    labels={'carat': 'Вес (караты)', 'price': 'Цена (USD)'},
    template='plotly_white'
)
st.plotly_chart(fig1, use_container_width=True)
st.caption("Наблюдается сильная положительная корреляция: чем больше вес, тем выше цена. Огранка Ideal часто дает более высокую цену при одинаковом весе.")

# 2. Box plot: price по cut
st.subheader("2. Распределение цены по категориям огранки")

fig2 = px.box(
    df, 
    x='cut', 
    y='price', 
    color='cut',
    title='Распределение цены в зависимости от качества огранки',
    labels={'cut': 'Огранка', 'price': 'Цена (USD)'},
    template='plotly_white'
)
st.plotly_chart(fig2, use_container_width=True)
st.caption("Огранка Ideal и Premium имеют более высокую медианную цену и меньший разброс.")

# 3. Heatmap корреляций
st.subheader("3. Матрица корреляций числовых признаков")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr_matrix = df[numeric_cols].corr()

fig3, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, 
    mask=mask,
    annot=True, 
    fmt='.2f', 
    cmap='RdBu_r',
    center=0,
    square=True,
    linewidths=0.5,
    ax=ax
)
ax.set_title('Матрица корреляций признаков')
st.pyplot(fig3)
st.caption("Сильная корреляция между carat, x, y, z и price. Признаки x, y, z сильно коррелируют между собой (мультиколлинеарность).")

# 4. Violin plot: price по color
st.subheader("4. Распределение цены по цвету (color)")

fig4 = px.violin(
    df, 
    x='color', 
    y='price', 
    color='color',
    box=True,
    title='Распределение цены в зависимости от цвета',
    labels={'color': 'Цвет (D - лучший, J - худший)', 'price': 'Цена (USD)'},
    template='plotly_white'
)
st.plotly_chart(fig4, use_container_width=True)
st.caption("Цвет D (бесцветный) имеет наибольшую цену. С ухудшением цвета (E→J) цена снижается.")

# 5. Дополнительная визуализация: гистограмма распределения цены
st.subheader("5. Гистограмма распределения целевой переменной (price)")

fig5, axes = plt.subplots(1, 2, figsize=(12, 5))

# Исходное распределение
axes[0].hist(df['price'], bins=50, edgecolor='black', alpha=0.7, color='skyblue')
axes[0].set_xlabel('Price (USD)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Исходное распределение price')

# Логарифмированное распределение (если есть log_price)
if 'log_price' in df.columns:
    axes[1].hist(df['log_price'], bins=50, edgecolor='black', alpha=0.7, color='salmon')
    axes[1].set_xlabel('Log(Price)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Распределение после логарифмирования')
else:
    # Создаем log распределение для демо
    log_price = np.log1p(df['price'])
    axes[1].hist(log_price, bins=50, edgecolor='black', alpha=0.7, color='salmon')
    axes[1].set_xlabel('Log(Price)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Распределение после логарифмирования')

plt.tight_layout()
st.pyplot(fig5)
st.caption("Логарифмическое преобразование делает распределение близким к нормальному, что улучшает качество регрессионных моделей.")

st.info("💡 **Инсайты:** Вес (carat) — самый важный фактор цены. Качество огранки, цвета и чистоты также значимо влияют на стоимость.")
