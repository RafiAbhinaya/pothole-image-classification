# import libraries
import numpy as np
import tensorflow as tf
import streamlit as st

# load model
model = tf.keras.models.load_model('final_model.keras')

# function for prediction
def prediction(file):
    img = tf.keras.utils.load_img(file, target_size=(256,256))

    x = tf.keras.utils.img_to_array(img)/255
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    y_pred_proba = model.predict(images)

    if y_pred_proba > 0.5:
        return 'Pothole'
    else:
        return 'Plain Road'

# run function
def run():
    # show title
    st.title('Pothole Detection🔍')
    st.caption('By Muhammad Rafi Abhinaya')

    # show description
    st.markdown('''
    In this section, you'll be able to input an image and the model will classify if it's a plain road or pothole.
    ''')

    # create form to contain data input
    image = st.file_uploader('Upload an image.', type=['jpg', 'jpeg', 'png'])

    if image is not None:
        # display uploaded image
        st.image(image, caption='Uploaded Image', use_container_width=True)

        # predict button
        if st.button('Classify'):
            result = prediction(image)
            if result == 'Plain Road':
                st.success('Result: There are no potholes on the road!')
            elif result == 'Pothole':
                st.error('Result: There are potholes on the road!')

# run if in file
if __name__ == '__main__':
    run()
