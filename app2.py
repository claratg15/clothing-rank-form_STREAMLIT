import streamlit as st
import os
import random
import pandas as pd
from streamlit_sortables import sort_items

st.set_page_config(
    layout="wide",
)

# Carpeta on es troben les imatges
IMAGE_FOLDER = "subset_100_images"  # Canvia-ho segons la ubicació real

# Arxiu on es desen les respostes
DATA_FILE = "responses.csv"

# 🌟 Mantenir les mateixes imatges durant la sessió
if "selected_images" not in st.session_state:
    def get_random_images(folder, n=10):
        images = [f for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        return random.sample(images, min(n, len(images)))

    st.session_state.selected_images = get_random_images(IMAGE_FOLDER)

# 🖼️ Llista d'imatges seleccionades
image_list = st.session_state.selected_images

# Personalitzar el color del quadre d'ordenació
st.markdown(
    """
    <style>
    /* Canviar el color de fons del quadre d'ordenació */
    .streamlit-sortables__item {
        background-color: #f0f8ff !important;  /* Color blau clar, canvia-ho pel que vulguis */
    }
    
    /* Canviar el color del text de les etiquetes */
    .streamlit-sortables__item-text {
        color: #333333 !important;  /* Color de text, pot ser negre o el que vulguis */
    }

    /* Si vols també canviar el color del quadre de selecció que mostra l'ordenació */
    .streamlit-sortables__container {
        background-color: #ffffff; /* Color de fons del contenidor */
    }
    </style>
    """, unsafe_allow_html=True
)


# Comprovar si les imatges ja han estat ordenades
if "submitted" in st.session_state and st.session_state.submitted:
    # Si ja s'ha enviat, mostrar el missatge de gràcies i no mostrar les imatges
    st.success("Moltes gràcies per enviar el teu ranking! 🎉")
else:
    st.title("🛍️ Enquesta de preferència de peces de roba")
    st.write("📌 **Arrossega les imatges per ordenar-les segons la teva preferència.**")

    # 🔢 Crear etiquetes "Imatge 1", "Imatge 2"...
    image_labels = [f"Imatge {i+1}" for i in range(len(image_list))]

    # Ordenació de les imatges (mostrant només els noms als sortables)
    sorted_filenames = sort_items(image_labels, direction="vertical")

    # Reordenar les imatges en funció de l'ordre de les etiquetes
    sorted_images = [os.path.join(IMAGE_FOLDER, image_list[image_labels.index(label)]) for label in sorted_filenames]

    # 📷 Mostrar les imatges ordenades
    st.write("🔽 **Ordre seleccionat:**")
    cols = st.columns(5)  # Mostrar les imatges en files de 5

    # Mostrar les imatges amb etiquetes reordenades
    for i, img in enumerate(sorted_images):
        with cols[i % 5]:  
            st.write(f"**{sorted_filenames[i]}**")  # Mostrar les etiquetes reordenades
            st.image(img, use_container_width=True)  # Utilitzar l'amplada del contenidor per les imatges

    # Funció per obtenir només el nom de la imatge sense l'extensió
    def get_image_name(image_path):
        return os.path.splitext(os.path.basename(image_path))[0]

    # Botó per desar la resposta
    if st.button("Desar resposta"):
        if sorted_images:
            # Obtenir els noms de les imatges sense la ruta ni l'extensió
            sorted_image_names = [get_image_name(img) for img in sorted_images]

            # Emmagatzemar el nom real de la imatge
            user_id = len(pd.read_csv(DATA_FILE)) + 1 if os.path.exists(DATA_FILE) else 1  # ID incremental
            df = pd.DataFrame([[user_id] + sorted_image_names], columns=["ID"] + [f"Rank_{i}" for i in range(1, len(sorted_image_names) + 1)])

            # Desa en CSV
            if os.path.exists(DATA_FILE):
                df.to_csv(DATA_FILE, mode='a', header=False, index=False)
            else:
                df.to_csv(DATA_FILE, index=False)

            # Marcar que les respostes ja han estat enviades
            st.session_state.submitted = True

            # Mostrar missatge de gràcies i ocultar imatges
            st.success("Moltes gràcies per enviar el teu ranking! 🎉")

# No mostrar l'opció per veure les respostes
# if st.checkbox("📊 Veure respostes guardades"):
#     if os.path.exists(DATA_FILE):
#         st.dataframe(pd.read_csv(DATA_FILE))
#     else:
#         st.write("Encara no hi ha respostes.")
