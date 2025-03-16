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

st.markdown(
    """
    <style>
    /* Canviar el color de fons del quadre d'ordenació */
    .streamlit-sortables__container {
        background-color: #f0f8ff;  /* Color blau clar, canvia-ho pel que vulguis */
    }
    
    /* Canviar el color del text de les etiquetes */
    .streamlit-sortables__item {
        color: #333333;  /* Color de text, pot ser negre o el que vulguis */
    }
    </style>
    """, unsafe_allow_html=True
)

st.title("Enquesta de preferència de peces de roba")
st.write("**Arrossega les imatges per ordenar-les segons la teva preferència. Les fotografies es troben a sota del requadre.**")

# 🔢 Crear etiquetes "Imatge 1", "Imatge 2"...
image_labels = [f"Imatge {i+1}" for i in range(len(image_list))]

# Ordenació de les imatges (mostrant només els noms als sortables)
sorted_filenames = sort_items(image_labels, direction="vertical")

# Reordenar les imatges en funció de l'ordre de les etiquetes
sorted_images = [os.path.join(IMAGE_FOLDER, image_list[image_labels.index(label)]) for label in sorted_filenames]

# 📷 Mostrar les imatges ordenades
st.write(" **Imatges segons l'ordre seleccionat:**")

# Mostrar les imatges amb etiquetes reordenades
cols = st.columns(5)  # Mostrar les imatges en files de 5

for i, img in enumerate(sorted_images):
    with cols[i % 5]:  
        st.write(f"**{sorted_filenames[i]}**")  # Mostrar les etiquetes reordenades
        st.image(img, use_container_width=True)  # Utilitzar l'amplada del contenidor per les imatges

# Funció per obtenir només el nom de la imatge sense l'extensió
def get_image_name(image_path):
    return os.path.splitext(os.path.basename(image_path))[0]

# Estilitzar el quadre drag-and-drop per fer-lo més petit
st.markdown(
    """
    <style>
    /* Reduir la mida del quadre drag-and-drop */
    .streamlit-sortable-items {
        max-width: 80% !important;  /* Ajusta el 80% de l'amplada de la pàgina */
        margin: 0 auto;  /* Centrar el quadre */
    }
    </style>
    """, unsafe_allow_html=True
)

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

        st.success("Resposta desada correctament. Moltes gràcies per la teva participació")
