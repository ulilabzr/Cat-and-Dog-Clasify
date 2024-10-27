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
    target_size=(64,64)
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = np.expand_dims(image_array,axis=0)
    image_array = image_array.astype('float32') / 255.0
    return image_array

def predict(model,image):
    prediction = model.predict(image)
    return prediction
    

def interpret_prediction(prediction):
    if prediction.shape[-1] == 1:
        score = prediction[0][0]
        predicted_class = 0 if score <= 0.5 else 1
        confidence_score = [score, 1- score,0 ]
    else:
        confidence_score = [0]
        predicted_class = np.argmacx(confidence_score)
    return predicted_class, confidence_score

def main():
    st.set_page_config(
    page_title="Your Page Title",
    layout="wide"
    )

    st.sidebar.header("Deskripsi")
    st.sidebar.text("ini adalah deskripsi")

    st.markdown(
    "<h1 style='text-align: center; color: #FF5733;'>Klasifikasi Anjing dan Kucing</h1>",
    unsafe_allow_html=True
)

    
    
    try:
        model = load_model()
        #st.sidebar("model output shape", model.output_shape)
    except Exception as err:
        st.error(f"error : {str(err)}")
        return
    
    uploader = st.file_uploader("PIlih gambar Anjing / Kucing", type=['jpg','jpeg','png'])


    if 'predictions' not in st.session_state:
        st.session_state.predictions = {'Kucing': 0, 'Anjing': 0}


    if uploader is not None:
        try:
            col1,col2 = st.columns([2,1])
            with col1:
                image = Image.open(uploader)
                st.image(image,caption="testing gambar", use_column_width=True)

            with col2:
                if st.button('classify',use_container_width=True):
                    with st.spinner('sedang menghitung'):
                        processed_image = preprocessing_image(image)
                        prediction = predict(model,processed_image)
                        predicted_class,confidence_score = interpret_prediction(prediction)
                        
                        class_names = ['Kucing','Anjing']
                        result = class_names[predicted_class]
                        st.success(f"Hasil Prediksi : {result.capitalize()}")

                        st.session_state.predictions[result.capitalize()] += 1
                        
                        confidence = confidence_score[0] * 100
                        st.progress(int(confidence))
                        st.write(f"Prosentase klasfikasi Anjing: {confidence:.2f}%")
                        confidence = confidence_score[1] * 100
                        st.progress(int(confidence))
                        st.write(f"Prosentase klasfikasi Kucing: {confidence:.2f}%")

                        st.header("Statistik Prediksi")
                        st.write(f"Kucing: {st.session_state.predictions['Kucing']}")
                        st.write(f"Anjing: {st.session_state.predictions['Anjing']}")

                        if st.button("Download Gambar"):
                            image.save(f"gambar_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                            st.success("Gambar berhasil diunduh.")
                #st.sidebar("model output shape", model.output_shape)

    

        except Exception as err:
            st.error(f"error : {str(err)}")
            st.write("Mohon Pilih file yang benar")
            st.write(f"Error ada di : {str(err)}")
            return

if __name__ == "__main__":
    main()