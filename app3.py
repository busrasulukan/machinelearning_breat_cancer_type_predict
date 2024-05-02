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
    data = get_clean_data()
    slider_labels ={"Teşhis":"Clusters: Malignant:1 / Benign: 0",
        "Ortalama Pürüzsüzlük": "Mean Smoothness",
        "Ortalama Kompaktlık": "Mean Compactness",
        "Ortalama Fraktal Boyut": "Mean Fractal Dimension",
        "SE Doku": "SE Texture",
        "SE Alan": "SE Area",
        "SE Pürüzsüzlük": "SE Smoothness",
        "SE Kompaktlık": "SE Compactness",
        "SE İçbükeylik": "SE Concavity",
        "SE İçbükey Noktalar": "SE Concave Points",
        "SE Simetri": "	SE Symmetry",
        "SE Fraktal Boyut": "SE Fractal Dimension",
        "En Kötü Durum Doku":"Worst Texture",
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
    # Modeli yükle
    model = pickle.load(open("newmodel.pkl", "rb"))
    scaler = pickle.load(open("newscaler.pkl", "rb"))
    input_np = np.array(list(input_data.values())).reshape(1, -1)
    input_scaled = scaler.transform(input_np)
    prediction = model.predict(input_scaled)
    # Tahmin sonucunu belirleme
    diagnosis = "Malignant" if prediction[0] == 1 else "Benign"
    benign_prob = round(model.predict_proba(input_scaled)[0][0], 3)
    malicious_prob = round(model.predict_proba(input_scaled)[0][1], 3)

    # Tahmin sonucunu gösterme
    st.markdown("<h2 style='color: #F08080;'>Cell Cluster Status Result 📍</h2>", unsafe_allow_html=True)
    if diagnosis == "Benign":
        st.markdown(f"<h3 style='color:green; font-size:20px;'>{diagnosis}</h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='color:red; font-size:20px;'>{diagnosis}</h3>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='font-size: 20px;'>Benign Probability:</h3>", unsafe_allow_html=True)
    st.write(benign_prob)
    st.markdown("<h3 style='font-size: 20px;'>Malicious Probability:</h3>", unsafe_allow_html=True)
    st.write(malicious_prob)


    st.write("The analysis is to purely boost the quality of diagnosis and is not meant as a substitute to professional diagnosis")

def main():
    # Sayfa Ayarları
    st.set_page_config(
        page_title="Breast Cancer Prediction App",
        page_icon="background.jpg",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Başlık Ekleme
    st.markdown("<h1 style='color: #F08080;'>🍀🌸 Breast Cancer Classification Project 🌸🍀</h1>", unsafe_allow_html=True)
    # Resim Ekleme
    st.image("background.jpg")

    # Markdown Oluşturma
    st.markdown("A research company wants to decide whether this cancer is in <font color='#e85a79'>**Benign**</font> or <font color='#e85a79'>**Malignant**</font> by looking at the various features of the breast cancers they have.", unsafe_allow_html=True)
    st.markdown("After the latest developments in the artificial intelligence industry, they expect us to develop a <font color='#e85a79'>**machine learning model**</font> in line with their needs and help them with their research.", unsafe_allow_html=True)
    st.markdown("In addition, when they have information about breast cancers, they want us to come up with a product that we can predict where this cancer will be based on this information.")
    st.markdown("*Let's help them!*")

    st.image("öneri.jpg")
    # Header Ekleme
    st.markdown("<h2 style='color: #F08080;'>Data Dictionary[EN]🎗️Veri Sözlüğü [TR]</h2>", unsafe_allow_html=True)
    data_dict_en = {
    "Mean Smoothness": "Mean of smoothness value for the cells",
    "Mean Compactness": "Mean of compactness value for the cells",
    "Mean Fractal Dimension": "Mean of fractal dimension value for the cells",
    "SE Texture": "Standard error of texture value for the cells",
    "SE Area": "Standard error of area value for the cells",
    "SE Smoothness": "Standard error of smoothness value for the cells",
    "SE Compactness": "Standard error of compactness value for the cells",
    "SE Concavity": "Standard error of concavity value for the cells",
    "SE Concave Points": "Standard error of concave points value for the cells",
    "SE Symmetry": "Standard error of symmetry value for the cells",
    "SE Fractal Dimension": "Standard error of fractal dimension value for the cells",
    "Worst Texture": "Worst (largest) value of texture for the cells",
    "Worst Area": "Worst (largest) value of area for the cells",
    "Worst Smoothness": "Worst (largest) value of smoothness for the cells",
    "Worst Concave Points": "Worst (largest) value of concave points for the cells",
    "Worst Symmetry": "Worst (largest) value of symmetry for the cells",
    "Worst Fractal Dimension": "Worst (largest) value of fractal dimension for the cells"
    }

    data_dict_tr = {
        "Ortalama Pürüzsüzlük": "Hücreler için pürüzsüzlük değerinin ortalaması",
        "Ortalama Kompaktlık": "Hücreler için kompaktlık değerinin ortalaması",
        "Ortalama Fraktal Boyut": "Hücreler için fraktal boyut değerinin ortalaması",
        "SE Doku": "Hücreler için doku değerinin standart hatası",
        "SE Alan": "Hücreler için alan değerinin standart hatası",
        "SE Pürüzsüzlük": "Hücreler için pürüzsüzlük değerinin standart hatası",
        "SE Kompaktlık": "Hücreler için kompaktlık değerinin standart hatası",
        "SE İçbükeylik": "Hücreler için içbükeylik değerinin standart hatası",
        "SE İçbükey Noktalar": "Hücreler için içbükey nokta değerinin standart hatası",
        "SE Simetri": "Hücreler için simetri değerinin standart hatası",
        "SE Fraktal Boyut": "Hücreler için fraktal boyut değerinin standart hatası",
        "En Kötü Durum Doku": "Hücreler için en kötü (en büyük) doku değeri",
        "En Kötü Durum Alanı": "Hücreler için en kötü (en büyük) alan değeri",
        "En Kötü Durum Pürüzsüzlük": "Hücreler için en kötü (en büyük) pürüzsüzlük değeri",
        "En Kötü Durum İçbükey Noktalar": "Hücreler için içbükey noktaların en kötü (en büyük) değeri",
        "En Kötü Durum Simetrisi": "Hücreler için en kötü (en büyük) simetri değeri",
        "En Kötü Durum Fraktal Boyutu": "Hücreler için fraktal boyutun en kötü (en büyük) değeri"
    }

# İngilizce veri tablosunu oluştur
    data_en_df = pd.DataFrame(data_dict_en.items(), columns=["Feature", "Description"])

# Türkçe veri tablosunu oluştur
    data_tr_df = pd.DataFrame(data_dict_tr.items(), columns=["Feature", "Description"])

# İki tabloyu yan yana göstermek için iki sütun oluştur
    col1, col2 = st.columns(2)

# İngilizce veri tablosunu ilk sütuna ekle
    with col1:
        st.markdown("<h3 style='color: #F08080;'>English Data Dictionary</h3>", unsafe_allow_html=True)
        st.table(data_en_df.style.set_properties(**{'background-color': '#fbecf3', 'color': '#435c76'}))

# Türkçe veri tablosunu ikinci sütuna ekle
    with col2:
        st.markdown("<h3 style='color: #F08080;'>Türkçe Veri Sözlüğü</h3>", unsafe_allow_html=True)
        st.table(data_tr_df.style.set_properties(**{'background-color': '#fbecf3', 'color': '#435c76'}))

    data = get_clean_data()
    input_data = add_sidebar(data)
    with st.container():
        st.markdown("<h2 style='color: #F08080;'>User Input Data 📥</h2>", unsafe_allow_html=True)
        st.table(pd.DataFrame(input_data, index=[0]).style.set_properties(**{'background-color': '#fbecf3', 'color': '#435c76'}))
    add_prediction(input_data)

if __name__ == "__main__":
    main()
