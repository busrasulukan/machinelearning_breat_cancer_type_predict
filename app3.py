import pandas as pd
import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

def get_clean_data():
    data = pd.read_csv("breast_cancer2.csv")
    data['Teşhis'] = data['Teşhis'].map({'Malignant': 1, 'Benign': 0})
    return data

def add_sidebar(data):
    st.sidebar.header("User Input Data")
    slider_labels = {
        "Teşhis": "Clusters: Malignant:1 / Benign: 0",
        "Ortalama Pürüzsüzlük": "Mean Smoothness",
        "Ortalama Kompaktlık": "Mean Compactness",
        "Ortalama Fraktal Boyut": "Mean Fractal Dimension",
        "SE Doku": "SE Texture",
        "SE Alan": "SE Area",
        "SE Pürüzsüzlük": "SE Smoothness",
        "SE Kompaktlık": "SE Compactness",
        "SE İçbükeylik": "SE Concavity",
        "SE İçbükey Noktalar": "SE Concave Points",
        "SE Simetri": "SE Symmetry",
        "SE Fraktal Boyut": "SE Fractal Dimension",
        "En Kötü Durum Doku": "Worst Texture",
        "En Kötü Durum Alanı": "Worst Area",
        "En Kötü Durum Pürüzsüzlük": "Worst Smoothness",
        "En Kötü Durum İçbükey Noktalar": "Worst Concave Points",
        "En Kötü Durum Simetrisi": "Worst Symmetry",
        "En Kötü Durum Fraktal Boyutu": "Worst Fractal Dimension"
    }

    input_dict = {}
    for key, label in slider_labels.items():
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
    return input_dict

def add_prediction(input_data):
    # Load the model and scaler
    with open("newmodel.pkl", "rb") as model_file, open("newscaler.pkl", "rb") as scaler_file:
        model = pickle.load(model_file)
        scaler = pickle.load(scaler_file)

    input_np = np.array(list(input_data.values())).reshape(1, -1)
    input_scaled = scaler.transform(input_np)
    prediction = model.predict(input_scaled)

    diagnosis = "Malignant" if prediction[0] == 1 else "Benign"
    benign_prob = round(model.predict_proba(input_scaled)[0][0], 3)
    malignant_prob = round(model.predict_proba(input_scaled)[0][1], 3)

    st.markdown("<h2 style='color: #F08080;'>Cell Cluster Status Result 📍</h2>", unsafe_allow_html=True)
    if diagnosis == "Benign":
        st.markdown(f"<h3 style='color:green; font-size:20px;'>{diagnosis}</h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='color:red; font-size:20px;'>{diagnosis}</h3>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='font-size: 20px;'>Benign Probability:</h3>", unsafe_allow_html=True)
    st.write(benign_prob)
    st.markdown("<h3 style='font-size: 20px;'>Malignant Probability:</h3>", unsafe_allow_html=True)
    st.write(malignant_prob)

    st.write("The analysis is to purely boost the quality of diagnosis and is not meant as a substitute to professional diagnosis")

def main():
    st.set_page_config(
        page_title="Breast Cancer Prediction App",
        page_icon="🎗️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("<h1 style='color: #F08080;'>🍀🌸 Breast Cancer Classification Project 🌸🍀</h1>", unsafe_allow_html=True)
    st.image("background.jpg")

    st.markdown("A research company wants to decide whether this cancer is in <font color='#e85a79'>**Benign**</font> or <font color='#e85a79'>**Malignant**</font> by looking at the various features of the breast cancers they have.", unsafe_allow_html=True)
    st.markdown("After the latest developments in the artificial intelligence industry, they expect us to develop a <font color='#e85a79'>**machine learning model**</font> in line with their needs and help them with their research.", unsafe_allow_html=True)
    st.markdown("In addition, when they have information about breast cancers, they want us to come up with a product that we can predict where this cancer will be based on this information.")
    st.markdown("*Let's help them!*")

    st.image("öneri.jpg")

    data = get_clean_data()
    input_data = add_sidebar(data)

    with st.container():
        st.markdown("<h2 style='color: #F08080;'>User Input Data 📥</h2>", unsafe_allow_html=True)
        st.table(pd.DataFrame(input_data, index=[0]).style.set_properties(**{'background-color': '#fbecf3', 'color': '#435c76'}))
    
    add_prediction(input_data)

if __name__ == "__main__":
    main()
