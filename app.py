import pandas as pd
import streamlit as st
from PIL import Image
from bokeh.models.widgets import Div
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
import base64



title = 'Churn Prediction and Analysis of Internet Service Provider'



# Layout
img = Image.open('assets/icon_pink-01.png')
st.set_page_config(page_title=title, page_icon=img, layout='wide')






st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.center {
  display: block;
  margin-left: auto;
  margin-right: auto;
#   width: 50%;
}
</style> """, unsafe_allow_html=True)


padding = 0
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)

file_name='style.css'
with open(file_name) as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)






# Content
@st.cache
def load_data():
    df_raw = pd.read_csv(r'data/internet_service_churn.csv')
    df = df_raw.copy()
    return df_raw, df

df_raw, df = load_data()

def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s" class="center" width="100" height="100"/>' % b64
    st.write(html, unsafe_allow_html=True)


# Sidebar color
st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #ef4da0;
    }
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    f = open("assets/icon-01.svg","r")
    lines = f.readlines()
    line_string=''.join(lines)

    render_svg(line_string)

    st.write('\n')
    st.write('\n')
    st.write('\n')

    if st.button('🏠 HOME', on_click='https://muarrikhyazka.github.io'):
        js = "window.location.href = 'https://muarrikhyazka.github.io'"  # Current tab
        html = '<img src onerror="{}">'.format(js)
        div = Div(text=html)
        st.bokeh_chart(div)

    if st.button('🍱 GITHUB', on_click='https://github.com/muarrikhyazka'):
        js = "window.location.href = 'https://github.com/muarrikhyazka'"  # Current tab
        html = '<img src onerror="{}">'.format(js)
        div = Div(text=html)
        st.bokeh_chart(div)








st.title(title)


st.subheader('Business Understanding')
st.write(
    """
    Churn is time when client stop using product or service from certain company. If company can predict when churn of client happen, so they can 
    arrange strategy to retain the client before the D-day of churn. This churn prediction will help company to retain the client and the revenue as well.
    """
)

st.write(
    """
    In this case, we take POV from Internet Service Provider Company and use their data.
    """
)

st.subheader('Data Understanding')
st.write(
    """
    **Source : [Kaggle.](https://www.kaggle.com/datasets/mehmetsabrikunt/internet-service-churn?sort=votes)**
    """
)

st.write(
    """
    **Below is sample of the data.** 
    """
)

st.dataframe(df.sample(5))

st.subheader('Method')
st.write(
    """
    Compared some method of category prediction modeling. They are :
    \n1. Logistic Regression
    \n2. K-Neighbors Classifier
    \n3. Decision Tree Classifier
    \n4. Random Forest Classifier
    \n5. XGBoost Classifier
    \n6. CatBoost Classifier
    """
)

st.write("""
    **Flowchart**
""")

graph = graphviz.Digraph()
graph.edge('EDA', 'Data Preprocessing')
graph.edge('Data Preprocessing', 'Modeling')
graph.edge('Modeling', 'Hyperparameter Tuning')



st.graphviz_chart(graph)

st.subheader('Exploratory Data Analysis')

num = list(df.describe().columns)
fig_1 = plt.figure(figsize=(15, 20))
for i in range(0, len(num)):
    plt.subplot(7, 2, i+1)
    sns.histplot(x=df[num[i]], color='cornflowerblue')
    plt.xlabel(num[i])
    plt.tight_layout()
st.pyplot(fig_1.figure)

st.write(
    """
    From this plot, we can see the distribution and balance of each variables.
    """
)

num = list(df.describe().columns)
fig_2 = plt.figure(figsize=(20,10))
for i in range(0, len(num)):
    plt.subplot(2, 7, i+1)
    sns.boxplot(y=df[num[i]], orient='v', x=df['churn'], palette="Set2")
    plt.xticks(rotation=-45, ha='left')
    plt.tight_layout()
st.pyplot(fig_2.figure)

st.write(
    """
    On this chart, we can see distribution of each variable classified by the churn. From here, we got a picture which variable might be have big influence to the churn.'
    """
)

fig_3 = plt.figure(figsize=(15, 15))
sns.heatmap(df.corr(), cmap="YlGnBu", annot=True)
st.pyplot(fig_3.figure)

st.write(
    """
    Here we can see correlation across each variables. From here we can know what variable influence churn.'
    """
)

st.subheader('Modeling')
st.write(
    """
    Before Modeling, I do some small tasks. They are :
    \n1. Split data train and test.
    \n2. Standardize data.
    """
)

st.write(
    """
    Below is the performance of each methods, from there we can choose which the best.
    """
)

@st.cache
def load_data_performance():
    df_raw = pd.read_csv(r'data/table_comparison.csv', sep=';')
    return df_raw

df_performance = load_data_performance()
st.dataframe(df_performance)

st.write(
    """
    I decide to choose Catboost because has the biggest rocauc score.
    """
)

st.write(
    """
    After that, do Hyperparameter Tuning Grid Search Optuna on Catboost model. Below the best parameter to use.
    """
)

st.code("""
Number of finished trials: 31
Best trial:
  Value: 0.9378314808836415
  Params: 
    objective: Logloss
    colsample_bylevel: 0.09949250593150287
    depth: 12
    boosting_type: Ordered
    bootstrap_type: MVS
""")

st.subheader('Insight')
st.write(
    """
    For Model interpretation, used Shapley Value.
    """
)

st.image(image = Image.open('chart/shapley value.png'), caption='Red : push client into churn | Blue : push client to avoid churn')



st.write(
    """
    From chart above, internet service provider should increase their download limit or provide special product with high download limit.
    Beside that, minimize service failure can minimize the churn as well, so needed mitigation for it.
    """
)

c1, c2 = st.columns(2)
with c1:
    st.info('**[Github Repo](https://github.com/muarrikhyazka/internet-service-provider-churn-prediction)**', icon="🍣")

