import streamlit as st
import pandas as pd
import joblib

# --- KONVERSI KURS GBP KE IDR ---
KURS_GBP_TO_IDR = 20000  # Ganti jika kurs berubah

# --- SETTING HALAMAN ---
st.set_page_config(page_title="Prediksi Harga Airbnb London", layout="centered")
st.title("💰 Prediksi Harga Sewa Airbnb London")
st.write(
    "Masukkan detail properti untuk memprediksi **kategori harga**, "
    "**estimasi harga per malam**, dan **total biaya untuk beberapa malam** dalam GBP dan Rupiah."
)

# --- MUAT MODEL ------------------------------------------------------------
@st.cache_resource
def load_models():
    model_kategori = joblib.load("model_kategori_harga.pkl")
    model_regresi = joblib.load("model_prediksi_harga.pkl")
    return model_kategori, model_regresi

model_kat, model_reg = load_models()

# --- INPUT USER ------------------------------------------------------------
room_type = st.selectbox("Jenis Kamar", ["Entire home/apt", "Private room", "Shared room"])
person_capacity = st.number_input("Kapasitas Tamu", min_value=1, max_value=20, value=2)
cleanliness_rating = st.slider("Rating Kebersihan", 1.0, 10.0, 8.0)
bedrooms = st.number_input("Jumlah Kamar Tidur", min_value=0, max_value=10, value=1)
dist = st.number_input("Jarak dari Pusat Kota (km)", min_value=0.0, max_value=30.0, value=3.0)
nights = st.number_input("Jumlah Malam Menginap", min_value=1, max_value=30, value=2)

# --- BUAT DATAFRAME DARI INPUT ---------------------------------------------
input_df = pd.DataFrame([{
    "room_type": room_type,
    "person_capacity": person_capacity,
    "cleanliness_rating": cleanliness_rating,
    "bedrooms": bedrooms,
    "dist": dist
}])

# --- PREDIKSI --------------------------------------------------------------
if st.button("Prediksi"):
    # Prediksi kategori harga
    kategori = model_kat.predict(input_df)[0]

    # Prediksi nominal harga per malam (GBP)
    harga_per_malam = model_reg.predict(input_df)[0]
    harga_per_malam_rounded = round(harga_per_malam, 2)

    # Total harga
    total_gbp = harga_per_malam * nights
    total_idr = round(total_gbp * KURS_GBP_TO_IDR)

    # Tampilkan hasil
    st.subheader("💡 Hasil Prediksi")
    st.success(f"Kategori harga: **{kategori}**")
    st.info(
        f"💷 Estimasi harga per malam: **£{harga_per_malam_rounded:,.2f}**\n\n"
        f"🛌 Estimasi total untuk {nights} malam:\n"
        f"**£{total_gbp:,.2f}**  (≈ Rp {total_idr:,.0f})"
    )
