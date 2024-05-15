from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import pandas as pd
import pickle
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_selection import SelectPercentile, chi2
import matplotlib.pyplot as plt
from joblib import load
import pickle
import re
from google_play_scraper import Sort, reviews
from google_play_scraper import app
from collections import Counter
import openpyxl

def scrape_reviews(app_id, count):
    result, _ = reviews(
        app_id,
        lang='id,en',
        country='id',
        sort=Sort.NEWEST, #memilih data yang terbaru
        count=count,
        filter_score_with=None
    )
    return result

best_model = load('best_model.pkl')

# Memuat TfidfVectorizer yang sama yang digunakan saat melatih model
with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

# Memuat selector yang sama yang digunakan saat melatih model
with open('selector.pkl', 'rb') as file:
    selector = pickle.load(file)

# Memuat objek lain yang dibutuhkan untuk preprocessing
stemmer = StemmerFactory().create_stemmer()

# Membaca daftar stopwords dari file Stopwords.txt
stop_words_file = 'Stopwords.txt'
with open(stop_words_file, 'r', encoding='utf-8') as file:
    stop_words = set([word.strip() for word in file.readlines() if word.strip()])

# Fungsi untuk preprocessing teks
def preprocess_text(text):
    tokens = text.split()  # Tokenisasi dengan split berdasarkan spasi
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    
    # Menghapus karakter berulang
    cleaned_tokens = []
    for token in filtered_tokens:
        cleaned_token = re.sub(r'(.)\1+', r'\1', token)
        cleaned_tokens.append(cleaned_token)
    
    stemmed_tokens = [stemmer.stem(word) for word in cleaned_tokens]
    processed_text = ' '.join(stemmed_tokens)
    return processed_text


# Fungsi untuk preprocessing teks
# def preprocess_text(text):
#     tokens = word_tokenize(text)
#     filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    
#     # Menghapus karakter berulang
#     cleaned_tokens = []
#     for token in filtered_tokens:
#         cleaned_token = re.sub(r'(.)\1+', r'\1', token)
#         cleaned_tokens.append(cleaned_token)
    
#     stemmed_tokens = [stemmer.stem(word) for word in cleaned_tokens]
#     processed_text = ' '.join(stemmed_tokens)
#     return processed_text



# Fungsi untuk prediksi sentimen dan mengembalikan prediksi serta probabilitasnya (TFIDF)
def predict_sentiment_with_prob(text):
    processed_text = preprocess_text(text)
    tfidf_text = tfidf_vectorizer.transform([processed_text])
    tfidf_text_selected = selector.transform(tfidf_text)
    prediction = best_model.predict(tfidf_text)
    probabilities = best_model.predict_proba(tfidf_text)
    return prediction[0], probabilities[0]

# Fungsi untuk memproses dan mendeteksi sentimen dari file Excel yang diunggah
def process_uploaded_file(uploaded_file):
    df = pd.read_excel(uploaded_file)
    results = []
    for index, row in df.iterrows():
        text = row['Ulasan']
        prediction, probabilities = predict_sentiment_with_prob(text)
        if prediction == 0:
            result = "Negatif"
        else:
            result = "Positif"
        results.append({"Ulasan": text, "Prediksi Sentimen": result, "Probabilitas Sentimen Negatif": probabilities[0], "Probabilitas Sentimen Positif": probabilities[1]})
    return pd.DataFrame(results)

# Fungsi untuk menghitung frekuensi kata berdasarkan label sentimen
def word_frequency_by_sentiment(df):
    positive_reviews = df[df['Prediksi Sentimen'] == 'Positif']['Ulasan'].tolist()
    negative_reviews = df[df['Prediksi Sentimen'] == 'Negatif']['Ulasan'].tolist()

    positive_words = []
    negative_words = []

    for review in positive_reviews:
        positive_words.extend(preprocess_text(review).split())

    for review in negative_reviews:
        negative_words.extend(preprocess_text(review).split())

    positive_word_freq = Counter(positive_words)
    negative_word_freq = Counter(negative_words)

    return positive_word_freq, negative_word_freq

# Fungsi untuk menampilkan diagram pie
def show_pie_chart(df):
    sentimen_counts = df['Prediksi Sentimen'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(sentimen_counts, labels=sentimen_counts.index, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    ax.set_title('Persentase Sentimen')
    st.pyplot(fig)

# Fungsi untuk menampilkan diagram batang frekuensi kata
def show_bar_chart(word_freq, title):
    common_words = word_freq.most_common(10)
    words = [word for word, freq in common_words]
    frequencies = [freq for word, freq in common_words]
    
    fig, ax = plt.subplots()
    ax.barh(words, frequencies, color='skyblue')
    ax.set_xlabel('Frekuensi')
    ax.set_title(title)
    plt.gca().invert_yaxis()
    st.pyplot(fig)

from io import BytesIO

# Menggunakan Streamlit untuk membuat antarmuka pengguna
def main():
    st.set_page_config(page_title="Sistem analisis sentimen ulasan")  # layout="wide" Menambahkan judul halaman dan layout wide
    # Header
    st.image("KLP.png", use_column_width=True)  # Menambahkan gambar header
    st.markdown(
        "<h1 style='text-align: center;'>Analisis Sentimen Ulasan Menggunakan Model Naive Bayes</h1>",
        unsafe_allow_html=True,
    ) 

    # Pilihan untuk input manual, upload file Excel, atau scraping data
    option = st.radio("Pilih Fitur:", ("Scraping Data", "Input Manual", "Upload File Excel"))
    if option == "Scraping Data":
        # Konten untuk scraping data
        app_id = st.text_input("Masukkan ID Aplikasi (App ID) Google Play Store. Misalnya https://play.google.com/store/apps/details?id=com.mobile.legends maka diambil hanya com.mobile.legends :")
        count = st.number_input("Masukkan Jumlah Ulasan yang Akan Diambil (Maksimal 400):", min_value=1, value=400)
        
        # if st.button("Mulai Scraping"):
        #     if app_id:
        #         st.write("Sedang melakukan scraping ulasan...")
        #         result = scrape_reviews(app_id, count)
        #         df_data = pd.DataFrame(result)
        #         df_data.rename(columns={'content':'Ulasan'}, inplace=True)  # Mengubah nama kolom
        #         df_data = df_data[['Ulasan']]  # Menampilkan kolom 'Ulasan' saja
        #         st.write("Total data scraping:", len(df_data))
        #         st.write(df_data)
        #         # display_and_download_data(df_data)
        #         # Tombol untuk mendownload hasil scraping dalam bentuk file Excel
        #         st.button("Download Hasil Scraping")
        #         excel_buffer = BytesIO()
        #         df_data.to_excel(excel_buffer, index=False)
        #         excel_buffer.seek(0)
        #         st.download_button(label="Klik untuk Download", data=excel_buffer, file_name="hasil_scraping_ulasan.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        #     else:
        #         st.warning("Masukkan ID Aplikasi Google Play Store terlebih dahulu.")
        
        # if st.button("Mulai Scraping"):
        #     if app_id:
        #         st.write("Sedang melakukan scraping ulasan...")
        #         result = scrape_reviews(app_id, count)
        #         df_data = pd.DataFrame(result)
        #         df_data.rename(columns={'content': 'Ulasan'}, inplace=True)  # Mengubah nama kolom
        #         df_data = df_data[['Ulasan']]  # Menampilkan kolom 'Ulasan' saja
        #         st.write("Total data scraping:", len(df_data))
        #         st.write(df_data)
        #         # display_and_download_data(df_data)
        #         # Tombol untuk mendownload hasil scraping dalam bentuk file Excel
        #         st.button("Download Hasil Scraping")
        #         excel_buffer = BytesIO()
        #         df_data.to_excel(excel_buffer, index=False)
        #         excel_buffer.seek(0)
        #         st.download_button(label="Klik untuk Download", data=excel_buffer, file_name="hasil_scraping_ulasan.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        #     else:
        #         st.warning("Masukkan ID Aplikasi Google Play Store terlebih dahulu.")
        if st.button("Mulai Scraping"):
            if app_id:
                st.write("Sedang melakukan scraping ulasan...")
                result = scrape_reviews(app_id, count)
                df_data = pd.DataFrame(result)
            
            # Memilih kolom 'content' dan mengubah namanya menjadi 'Ulasan'
                df_data = df_data[['content']]
                df_data.rename(columns={'content': 'Ulasan'}, inplace=True)
            
                st.write("Total data scraping:", len(df_data))
                st.write(df_data)
            
                    # Tombol untuk mendownload hasil scraping dalam bentuk file Excel
                    if st.button("Download Hasil Scraping"):
                    excel_buffer = BytesIO()
                    df_data.to_excel(excel_buffer, index=False)
                    excel_buffer.seek(0)
                    st.download_button(label="Klik untuk Download", data=excel_buffer, file_name="hasil_scraping_ulasan.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                st.warning("Masukkan ID Aplikasi Google Play Store terlebih dahulu.")
        
    elif option == "Input Manual":
        user_input = st.text_input("Masukkan ulasan tentang aplikasi:")
        if st.button("Prediksi Sentimen"):
            if user_input:
                prediction, probabilities = predict_sentiment_with_prob(user_input)
                if prediction == 0:
                    st.write("Hasil prediksi sentimen: Negatif")
                    st.snow()
                else:
                    st.write("Hasil prediksi sentimen: Positif")
                    st.balloons()     
                st.write("Probabilitas sentimen negatif:", f"{probabilities[0]*100:.2f}%")
                st.write("Probabilitas sentimen positif:", f"{probabilities[1]*100:.2f}%")
            else:
                    st.warning("Silakan masukkan ulasan tentang aplikasi terlebih dahulu.")

    elif option == "Upload File Excel":
        # Konten untuk upload file Excel
        uploaded_file = st.file_uploader("Unggah file Excel", type=["xls", "csv", "xlsx"])
        
        if uploaded_file is not None:
            # Memproses file yang diunggah dan menampilkan hasilnya
            st.write("Hasil Analisis Sentimen:")
            results_df = process_uploaded_file(uploaded_file)
            st.write(results_df)
            st.balloons()
            
            # Menampilkan diagram pie
            show_pie_chart(results_df)
            
            # Menghitung frekuensi kata berdasarkan sentimen
            positive_word_freq, negative_word_freq = word_frequency_by_sentiment(results_df)
            
            # Menampilkan diagram batang untuk kata-kata positif dan negatif yang paling sering muncul
            st.write("Frekuensi Kata Positif yang Paling Sering Muncul:")
            show_bar_chart(positive_word_freq, "Frekuensi Kata Positif yang Paling Sering Muncul")

            st.write("Frekuensi Kata Negatif yang Paling Sering Muncul:")
            show_bar_chart(negative_word_freq, "Frekuensi Kata Negatif yang Paling Sering Muncul")
            
            # Tombol untuk mendownload hasil prediksi dalam bentuk file Excel
            if st.button("Download Hasil Prediksi"):
                excel_buffer = BytesIO()
                results_df.to_excel(excel_buffer, index=False)
                excel_buffer.seek(0)
                st.download_button(label="Klik untuk Download", data=excel_buffer, file_name="hasil_prediksi_sentimen.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # Footer
    footer = """
    <style>
    .footer {
        # position: fixed;
        bottom: 0;
        width: 100%;
        background-color: #f8f9fa;
        padding: 10px 0;
        text-align: center;
        font-size: 12px;
        color: #6c757d;
    }
    </style>
    <div class="footer">
        <p>© 2024 KELOMPOK C1 - PPDM INFORMATIKA UDAYANA. All rights reserved.</p>
    </div>
    """
    st.markdown(footer, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
