#import library
import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st
import datetime

def load_model():
    model = tf.keras.models.load_model('intermediate_model.keras')
    return model

def preprocessing_image(image):
    target_size = (64, 64)
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array.astype('float32') / 255.0
    return image_array

def predict(model, image):
    prediction = model.predict(image)
    return prediction

def interpret_prediction(prediction):
    if prediction.shape[-1] == 1:
        score = prediction[0][0]
        predicted_class = 0 if score <= 0.5 else 1
        confidence_score = [score, 1 - score, 0]
    else:
        confidence_score = prediction[0]
        predicted_class = np.argmax(confidence_score)
    return predicted_class, confidence_score

def main():
    st.set_page_config(
        page_title="Klasifikasi Anjing dan Kucing",
        layout="wide"
    )

    st.sidebar.header("🌟 About")
    st.sidebar.markdown("Selamat datang di aplikasi klasifikasi anjing dan kucing! Aplikasi ini dirancang untuk membantu Anda dalam mengidentifikasi dan mengklasifikasikan gambar anjing dan kucing menggunakan teknologi kecerdasan buatan (AI).")
    st.sidebar.markdown("\n\nBagaimana Cara Kerja? \n1. Unggah Gambar: Pilih gambar anjing atau kucing dari perangkat Anda.\n\n2. Proses Klasifikasi: Aplikasi akan memproses gambar menggunakan model pembelajaran mendalam (deep learning) yang telah dilatih untuk mengenali fitur khas anjing dan kucing.\n\n3. Lihat Hasil: Setelah pemrosesan selesai, Anda akan melihat hasil prediksi lengkap dengan tingkat kepercayaan untuk setiap kategori.")

    st.markdown(
        "<h1 style='text-align: center; color: #FF5733;'>Klasifikasi Anjing dan Kucing</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <style>
        body {
            background-color: #f0f2f5;
            color: #333;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    try:
        model = load_model()
    except Exception as err:
        st.error(f"error : {str(err)}")
        return

    uploader = st.file_uploader("Pilih gambar Anjing / Kucing", type=['jpg', 'jpeg', 'png'])

    if 'predictions' not in st.session_state:
        st.session_state.predictions = {'Kucing': 0, 'Anjing': 0}

    if uploader is not None:
        try:
            col1, col2 = st.columns([2, 1])
            with col1:
                image = Image.open(uploader)
                st.image(image, caption="Gambar yang diunggah", use_container_width=True)

            with col2:
                if st.button('Classify', use_container_width=True):
                    with st.spinner('Sedang menghitung...'):
                        processed_image = preprocessing_image(image)
                        prediction = predict(model, processed_image)
                        predicted_class, confidence_score = interpret_prediction(prediction)

                        class_names = ['Kucing', 'Anjing']
                        result = class_names[predicted_class]
                        st.success(f"Hasil Prediksi 🔎 : {result.capitalize()}{predicted_class}{confidence_score}")

                        st.session_state.predictions[result.capitalize()] += 1
                        
                        kucing_confidence = confidence_score[1] * 100
                        anjing_confidence = confidence_score[0] * 100
                        st.progress(int(kucing_confidence))
                        st.write(f"Prosentase klasifikasi Kucing: {kucing_confidence:.2f}%")
                        st.progress(int(anjing_confidence))
                        st.write(f"Prosentase klasifikasi Anjing: {anjing_confidence:.2f}%")

                        st.header("Jumlah Hasil Prediksi")
                        st.write(f"😺 Kucing: {st.session_state.predictions['Kucing']}")
                        st.write(f"🐶 Anjing: {st.session_state.predictions['Anjing']}")

                        feedback = st.radio("Apakah klasifikasi ini akurat?", ("Ya", "Tidak"))
                        if feedback == "Tidak":
                            user_feedback = st.text_area("Berikan umpan balik atau saran:")
                            if user_feedback:
                                st.write("Terima kasih atas umpan balik Anda!")
                                st.write(f"Umpan balik Anda: {user_feedback}")

                        if st.button("Reset Statistik"):
                            st.session_state.predictions = {'Kucing': 0, 'Anjing': 0}
                            st.success("Statistik berhasil direset.")

                        if st.button("Download Gambar"):
                            image.save(f"gambar_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                            st.success("Gambar berhasil diunduh.")

        except Exception as err:
            st.error(f"error : {str(err)}")
            st.write("Mohon pilih file yang benar.")
            return

if __name__ == "__main__":
    main()
