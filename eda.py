# import libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# define img size and path
img_size = 256
train_path = 'dataset/train/'

# load train images from directory
datagen = tf.keras.preprocessing.image.ImageDataGenerator()
data = datagen.flow_from_directory(directory=train_path,
                                   target_size=(img_size,img_size),
                                   class_mode='binary',
                                   batch_size=32,
                                   shuffle=False)

# fucntion to run file
def run():
    ###
    # TITLE SECTION
    ###
    # title and desc
    st.title('Exploratory Data Analysis📊')
    st.caption('By Muhammad Rafi Abhinaya')
    st.markdown("""
        In this section we'll do a simple exploration on the images used for training the model.
    """)

    # divider
    st.divider()

    ###
    # CLASS PROPORTIONS
    ###

    # title and desc
    st.header('**Class Proportions**')
    st.markdown(
        "First, let's see the proportion of classes in our dataset."
    )

    # count amount in each class
    label_amount = np.bincount(data.labels)

    # class names
    labels = data.class_indices.keys()

    # create figure, divide to 3 ax and only use the middle one
    fig, axs = plt.subplots(1, 3, figsize=(10,5), facecolor='none', gridspec_kw={'width_ratios': [1, 2, 1]})
    # create pie chart
    wedges, texts, autotexts = axs[1].pie(
            x=label_amount,
            labels=labels, 
            colors=sns.color_palette('flare', n_colors=2),
            autopct='%1.1f%%',
            startangle=90,
            textprops={'color': 'white', 'size':11.5},
            wedgeprops={'edgecolor': 'white'})
    # set style
    axs[1].set_title('Proportion of Images', color='white', fontweight='bold')
    axs[0].axis('off')
    axs[2].axis('off')
    fig.patch.set_facecolor("#FFFFFF05")
    # show chart
    st.pyplot(fig)

    # description
    st.markdown('''
    From the result above, we can see that the proportion of plain road images and pothole images is very similair at about 50%. This means the dataset has a
    balanced amount of plain road images and pothole images.
    ''')

    # divider
    st.divider()

    ###
    # PLAIN
    ###

    # title and desc
    st.header('**Plain Road Images**')
    st.markdown(
        "Next, let's view the plain road images."
    )

    # create figure
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(25,12))
    n = 0

    # show images from plain class
    for i in range(2):
        for j in range(4):
            img = data[0][0][n].astype('uint8')
            ax[i][j].imshow(img)
            ax[i][j].set_title('Class - Plain')
            n += 1
    
    st.pyplot(fig)

    st.markdown("""
    The pictures are downloaded from google images, which means the source of the picture may differ from one another. This results plain road images that has
    different shot angles and various locations. From the pictures avoce we can also see that vehicles may be present in some images and some may not be present.
    But one certain thing is that all of these images does have roads that doesn't have potholes in them. 

    The advantage of this randomness in source is we get all kinds of road images that can help train the model to generalize better to all kinds of roads.
    When deployed later, the model will classify images obtained from CCTV that comes in various angles, so the variations of the picture in our dataset is beneficial.
    """)

    st.divider()

    # title and desc
    st.header('**Pothole Images**')
    st.markdown(
        "Now, let's view the pothole images."
    )

    # create figure
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(25,12))
    n = 0

    # show images from pothole class
    for i in range(2):
        for j in range(4):
            img = data[15][0][n].astype('uint8')
            ax[i][j].imshow(img)
            ax[i][j].set_title('Class - Plain')
            n += 1
    
    st.pyplot(fig)

    st.markdown("""
    Same as before, the source of the pictures varies from one another, resulting in various variation of potholes. From the pictures above, we can see that our
    dataset contains different type of potholes, some are just tiny patches of hole, while some are large branching potholes. Vehicles may also be present or not
    in these pictures. Eventhough the pictures are different, they all seem to show roads with potholes as intended.

    This variation is also beneficial to the model. In the real world, potholes have different shape and sizes. Having variations of pothole images can help our
    model learn various shape and sizes of potholes and not memorize just a certain pattern.
    """)

# run if in file
if __name__ == '__main__':
    run()
