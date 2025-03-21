import streamlit as st
import os
import random
import pandas as pd
from streamlit_sortables import sort_items
from pymongo import MongoClient


# 🔐 Leer la URI de MongoDB desde Streamlit Secrets
MONGO_URI = st.secrets["mongo"]["uri"]
DB_NAME = "TFG"
COLLECTION_NAME = "respostes"

# 🔗 Conectar a MongoDB
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

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

# Marca si la resposta ha estat desada per evitar noves respostes
if "response_saved" not in st.session_state:
    st.session_state.response_saved = False

st.title("Enquesta de preferència de peces de roba")
st.write("**Arrossega les imatges per ordenar-les segons la teva preferència. Les fotografies es troben a sota del requadre.**")

# 🔢 Crear etiquetes "Imatge 1", "Imatge 2"...
image_labels = [f"Imatge {i+1}" for i in range(len(image_list))]

# Ordenació de les imatges (mostrant només els noms als sortables)
sorted_filenames = sort_items(image_labels, direction="vertical")

# Reordenar les imatges en funció de l'ordre de les etiquetes
sorted_images = [os.path.join(IMAGE_FOLDER, image_list[image_labels.index(label)]) for label in sorted_filenames]

# 📷 Mostrar les imatges ordenades només si la resposta no ha estat desada
if not st.session_state.response_saved:
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

# 💾 Guardar a MongoDB
if not st.session_state.response_saved:
    if st.button("Desar resposta"):
        if sorted_images:
            sorted_image_names = [get_image_name(img) for img in sorted_images]

            # ID del usuario (opcional, podrías usar un timestamp en su lugar)
            user_id = collection.count_documents({}) + 1  

            # Guardar en MongoDB
            collection.insert_one({
                "user_id": user_id,
                "sorted_images": sorted_image_names
            })

            st.success("Resposta desada correctament a MongoDB. Moltes gràcies!")

            # Evitar respuestas duplicadas
            st.session_state.response_saved = True
            st.session_state.selected_images = []
else:
    st.write("Ja has respost l'enquesta. Moltes gràcies per participar!")
