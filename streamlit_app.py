import streamlit as st
from sklearn.datasets import load_wine, load_breast_cancer, load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# Custom CSS for styling with colors
page_bg_color = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-color: #1a1a1a;
    background-size: 180%;
    background-position: top left;
    background-repeat: no-repeat;
    background-attachment: local;
    padding-bottom: 120px;  /* Added padding to prevent overlap with footer */
}

[data-testid="stSidebar"] > div:first-child {
    background-color: #2a2a2a;
    background-position: center; 
    background-repeat: no-repeat;
    background-attachment: fixed;
}

[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

[data-testid="stToolbar"] {
    right: 2rem;
}

footer {visibility: hidden;}
.footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background-color: #2a2a2a;
    text-align: center;
    padding: 10px;
    font-size: 14px;
    color: #f0f0f0;
    z-index: 100;  /* Ensure footer is on top */
}
.footer a {
    color: #f97316;
    text-decoration: none;
}
.footer a:hover {
    color: #ea580c;
}
</style>
<div class="footer">
    <p>Developed with ❤️ by <a href="https://zaintheanalyst.com" target="_blank">Zain Haidar</a></p>
    <p>
        <a href="https://github.com/zainhaidar16" target="_blank">GitHub</a> |
        <a href="https://www.linkedin.com/in/zain-haidar" target="_blank">LinkedIn</a> |
        <a href="mailto:contact@zaintheanalyst.com">Email</a>
    </p>
</div>
"""


st.markdown(page_bg_color, unsafe_allow_html=True)

# Function to get the classifier based on user selection
def getClassifier(classifier):
    if classifier == 'SVM':
        c = st.sidebar.slider(label='Choose value of C', min_value=0.0001, max_value=10.0, value=1.0)
        model = SVC(C=c)
    elif classifier == 'KNN':
        neighbors = st.sidebar.slider(label='Choose Number of Neighbors', min_value=1, max_value=20, value=5)
        model = KNeighborsClassifier(n_neighbors=neighbors)
    else:
        max_depth = st.sidebar.slider('Max Depth', 2, 10, value=5)
        n_estimators = st.sidebar.slider('Number of Estimators', 1, 100, value=10)
        model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=1)
    return model

# Function to apply PCA and return the dataframe
def getPCA(df):
    pca = PCA(n_components=3)
    result = pca.fit_transform(df.loc[:, df.columns != 'Type'])
    df['pca-1'] = result[:, 0]
    df['pca-2'] = result[:, 1]
    df['pca-3'] = result[:, 2]
    return df

# Function to load the selected dataset
def return_data(dataset):
    if dataset == 'Wine':
        data = load_wine()
    elif dataset == 'Iris':
        data = load_iris()
    else:
        data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names, index=None)
    df['Type'] = data.target
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=1, test_size=0.2)
    return X_train, X_test, y_train, y_test, df, data.target_names

# App title with styling
st.markdown("<h1 style='text-align: center; color: #f97316;'>Multi-Model Classification with PCA</h1>", unsafe_allow_html=True)

# Description
st.markdown("<p style='text-align: center; color: #f0f0f0;'>Choose a Dataset and a Classifier in the sidebar. Input your values and get a prediction</p>", unsafe_allow_html=True)

# Sidebar selections
sideBar = st.sidebar
dataset = sideBar.selectbox('Select Dataset', ('Wine', 'Breast Cancer', 'Iris'))
classifier = sideBar.selectbox('Select Classifier', ('SVM', 'KNN', 'Random Forest'))

# Load and preprocess the data
X_train, X_test, y_train, y_test, df, classes = return_data(dataset)

# Display a sample of the dataframe
st.markdown("<h3 style='text-align: center; color: #f97316;'>Sample of Selected Dataset</h3>", unsafe_allow_html=True)
st.dataframe(df.sample(n=5, random_state=1).style.set_properties(**{'background-color': '#1a1a1a', 'color': '#f0f0f0'}))

# Display class names
st.subheader("Classes")
for idx, value in enumerate(classes):
    st.markdown(f"<span style='color: #f0f0f0;'>{idx}: {value}</span>", unsafe_allow_html=True)

# Apply PCA for visualization
df = getPCA(df)

# Create two columns for side-by-side plots
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"<h3 style='text-align: center; color: #f0f0f0;'>2D PCA Visualization - {dataset}</h3>", unsafe_allow_html=True)
    fig = plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x="pca-1", y="pca-2",
        hue="Type",
        palette=sns.color_palette("hls", len(classes)),
        data=df,
        legend="full"
    )
    plt.xlabel('PCA One')
    plt.ylabel('PCA Two')
    st.pyplot(fig)

with col2:
    st.markdown(f"<h3 style='text-align: center; color: #f0f0f0;'>3D PCA Visualization - {dataset}</h3>", unsafe_allow_html=True)
    fig2 = plt.figure(figsize=(8, 6)).add_subplot(111, projection='3d')
    fig2.scatter(
        xs=df["pca-1"],
        ys=df["pca-2"],
        zs=df["pca-3"],
        c=df["Type"],
    )
    fig2.set_xlabel('PCA One')
    fig2.set_ylabel('PCA Two')
    fig2.set_zlabel('PCA Three')
    st.pyplot(fig2.get_figure())

# Train the model and calculate scores
model = getClassifier(classifier)
model.fit(X_train, y_train)
test_score = round(model.score(X_test, y_test), 2)
train_score = round(model.score(X_train, y_train), 2)

# Display scores in the sidebar with model and dataset names
sideBar.subheader(f'{classifier} Model on {dataset} Dataset')
sideBar.markdown(f"<span style='color: #f0f0f0;'>Train Score: {train_score}</span>", unsafe_allow_html=True)
sideBar.markdown(f"<span style='color: #f0f0f0;'>Test Score: {test_score}</span>", unsafe_allow_html=True)

st.markdown("<p style='text-align: center; color: #f0f0f0;'>This app demonstrates the use of PCA and different classifiers on various datasets.</p>", unsafe_allow_html=True)
