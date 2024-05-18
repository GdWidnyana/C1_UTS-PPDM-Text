Penyelesaian Kasus 1 – [Analisis Sentimen Ulasan Aplikasi SATUSEHAT Mobile Menggunakan Algoritma Multinomial Naïve Bayes Classifier dan Seleksi Fitur Chi-Square”] 
Mata Kuliah		: Pengantar Pemrosesan Data Multimedia
Program Studi	: Informatika 
Fakultas		  : FMIPA 
Universitas		: Universitas Udayana 
Tahun Ajaran	: 2023/2024
==============================================================

Deskripsi Tugas:

Project ini merupakan implementasi dari kecerdasan buatan salah satunya adalah natural language processing (NLP) dengan kasus analisis sentimen. Project ini membangun sebuah model machine learning yang diperuntukan untuk menganalisa sentimen secara otomatis sehingga lebih hemat waktu, tenaga, dan biaya. Model pada project ini dibangun menggunakan dataset primer yang ditambang dengan metode scrapping pada aplikasi SATUSEHAT Mobile dan diperoleh 4953 data. Kemudian data tersebut dibersihkan dan diseimbangkan distribusi data sehingga di peroleh 2020 data dengan pelabelan dilakukan dalam 2 kelas (positif dan negatif) dengan metode manual. Setelah data seimbang dilanjutkan dengan tahap text pre-processing yang mencakup case folding, cleaning dan tokenization, normalisasi, stopwords, steeming sehingga diperoleh data sebanyak 1916 yang bersih dan siap untuk dilakukan modelisasi algoritma machine learning.  Selanjutnya dilakukan ekstraksi fitur TF-IDF dan seleksi fitur chi-square. Algoritma yang digunakan adalah Multinomial Naive Bayes dengan seleksi fitur sebanyak 40% mendapatkan akurasi di 83% (sebelum tuning) dan 96% setelah tuning hyperparamater GridsearchCv (alpha : 0.0001 dan fit prior : False). Model tersebut dideploy dalam bentuk website menggunakan frameword streamlit. Project ini diharapkan dapat menganalisa kualitas sebuah produk dengan data ulasan yang jumlahnya besar, sehingga dapat memberikan insight bagi perusahaan.

Link Demo Sistem : https://c1-uts-ppdm-sentimen-ulasan.streamlit.app/

Anggota kelompok:
1.	Ida Bagus Putu Ryan Paramasatya Putra	(2208561010)
2.	I Gede Widnyana					              (2208561016)
3.	I Gede Widiantara Mega Saputra		    (2208561022)
4.	Bayu Yudistira Ramadhan			          (2208561085)
Kelas : C
Pengampu Mata Kuliah:
Dr. AAIN Eka Karyawati, S.Si., M.Eng. https://udayananetworking.unud.ac.id/lecturer/2372-anak-agung-istri-ngurah-eka-karyawati

Supporting by Lab Ergonomic Computing PS Informatika FMIPA Unud https://if.unud.ac.id/
