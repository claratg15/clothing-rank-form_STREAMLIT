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
DATA_FILE = "responses2.csv"

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

# Botó per desar la resposta només si no s'ha desat una resposta anterior
if not st.session_state.response_saved:
    if st.button("Desar resposta"):
        if sorted_images:
            # Obtenir els noms de les imatges sense la ruta ni l'extensió
            sorted_image_names = [get_image_name(img) for img in sorted_images]

            # Crear les respostes en el nou format (10 files per persona)
            responses = []
            for rank, image_name in enumerate(sorted_image_names, start=1):
                responses.append([len(pd.read_csv(DATA_FILE)) + 1 if os.path.exists(DATA_FILE) else 1, image_name, rank])

            # Crear un DataFrame amb les noves respostes
            df = pd.DataFrame(responses, columns=["ID", "Nom_imatge", "Rank"])

            # Desa les respostes en CSV
            if os.path.exists(DATA_FILE):
                df.to_csv(DATA_FILE, mode='a', header=False, index=False)
            else:
                df.to_csv(DATA_FILE, index=False)

            st.success("Resposta desada correctament. Moltes gràcies per la teva participació!")

            # Marcar que la resposta ha estat desada i amagar les imatges
            st.session_state.response_saved = True
            st.session_state.selected_images = []  # Eliminar les imatges per evitar que es mostrin de nou
else:
    st.write("Ja has respost l'enquesta. Moltes gràcies per participar!")

