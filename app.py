# import libraries
import streamlit as st
from streamlit_option_menu import option_menu
import eda, prediction

# create sidebar
with st.sidebar:
    # create option menu
    page = option_menu(
        menu_title='Navigation',
        options=['Home', 'EDA', 'Inference'],
        icons=['house', 'bar-chart', 'gear-fill'],
        menu_icon='geo',
        default_index=0
    )

# if home (default)
if page == "Home":
    # title and stuff
    st.title("Pothole Image Classification🕳️")
    st.caption('By Muhammad Rafi Abhinaya')
    st.image("https://www.avisenlegal.com/wp-content/uploads/2023/04/shutterstock_1134687086-750x375.jpg", use_container_width=True)

    # desc
    st.markdown("""
    Potholes are a common issue on roads worldwide, caused by factors such as weather changes, heavy traffic, and poor maintenance. They not only lead to
    discomfort for drivers and passengers but also pose serious safety risks, potentially causing accidents or vehicle damage. In contrast, plain or undamaged
    roads provide a smooth and safe driving experience. Being able to accurately distinguish between potholes and plain road surfaces is crucial for modern
    infrastructure monitoring. Developing a model that can automatically classify road conditions from images helps in early detection and timely repair of potholes.
    This kind of automated system can support road maintenance efforts, reduce manual inspection time, improve road safety, and optimize resource allocation for repairs.
    """)
    
    # divider
    st.divider()

    # desc
    st.markdown("""
    In this app, I've trained and deployed a model that's able to classify between plain road images and pothole images. Feel free to try the model out through the
    navigation panel!
                
    The dataset used in this project is obtained from Kaggle and can be acessed [here](https://www.kaggle.com/datasets/virenbr11/pothole-and-plain-rode-images).
    """)

# if eda
elif page == "EDA":
    eda.run()

# if inference
elif page == "Inference":
    prediction.run()

# xreate spacer to push contact info to bottom
st.sidebar.markdown("---")
spacer = st.sidebar.empty()  # acts as a flexible spacer
for _ in range(10):
    spacer.text("")  # add blank lines to push content downward

# contact info
st.sidebar.markdown("Feel free to reach out to me through my contact links below!", unsafe_allow_html=True)

# contact links
st.sidebar.link_button('📧 Email',
                       url='mailto:mr.abhinaya26@gmail.com',
                       use_container_width=True)
st.sidebar.link_button('💼 LinkedIn',
                       url='https://www.linkedin.com/in/RafiAbhinaya/',
                       use_container_width=True)
st.sidebar.link_button('🐙 GitHub',
                       url='https://github.com/RafiAbhinaya',
                       use_container_width=True)