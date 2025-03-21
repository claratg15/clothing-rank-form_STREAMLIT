import streamlit as st
import os
import random
import pandas as pd
import dropbox
from streamlit_sortables import sort_items

# Configuración de la página
st.set_page_config(layout="wide")

# Carpeta donde se encuentran las imágenes
IMAGE_FOLDER = "subset_100_images"  # Cambia esto a la ubicación real

# Nombre del archivo en Dropbox donde se guardarán las respuestas
DATA_FILE = "/encuesta_respuestas.csv"  # Ruta del archivo en Dropbox

# 🌟 Mantener las mismas imágenes durante la sesión
if "selected_images" not in st.session_state:
    def get_random_images(folder, n=10):
        images = [f for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        return random.sample(images, min(n, len(images)))

    st.session_state.selected_images = get_random_images(IMAGE_FOLDER)

# 🖼️ Lista de imágenes seleccionadas
image_list = st.session_state.selected_images

# Marca si la respuesta ha sido guardada para evitar nuevas respuestas
if "response_saved" not in st.session_state:
    st.session_state.response_saved = False

# Título de la encuesta
st.title("Encuesta de preferencia de piezas de ropa")
st.write("**Arrastra las imágenes para ordenarlas según tu preferencia. Las fotografías están abajo.**")

# 🔢 Crear etiquetas "Imagen 1", "Imagen 2"...
image_labels = [f"Imagen {i+1}" for i in range(len(image_list))]

# Ordenación de las imágenes (mostrando solo los nombres en los elementos ordenables)
sorted_filenames = sort_items(image_labels, direction="vertical")

# Reordenar las imágenes en función del orden de las etiquetas
sorted_images = [os.path.join(IMAGE_FOLDER, image_list[image_labels.index(label)]) for label in sorted_filenames]

# 📷 Mostrar las imágenes ordenadas solo si la respuesta no ha sido guardada
if not st.session_state.response_saved:
    st.write(" **Imágenes según el orden seleccionado:**")

    # Mostrar las imágenes en columnas (5 imágenes por fila)
    cols = st.columns(5)

    for i, img in enumerate(sorted_images):
        with cols[i % 5]:  
            st.write(f"**{sorted_filenames[i]}**")  # Mostrar las etiquetas reordenadas
            st.image(img, use_container_width=True)  # Utilizar el ancho del contenedor para las imágenes

# Función para obtener solo el nombre de la imagen sin la extensión
def get_image_name(image_path):
    return os.path.splitext(os.path.basename(image_path))[0]

# Función para subir el archivo CSV a Dropbox
def upload_to_dropbox(file_path, dropbox_path, token):
    dbx = dropbox.Dropbox(token)
    
    with open(file_path, "rb") as f:
        dbx.files_upload(f.read(), dropbox_path, mode=dropbox.files.WriteMode.overwrite)

# Botón para guardar la respuesta
if not st.session_state.response_saved:
    if st.button("Guardar respuesta"):
        if sorted_images:
            # Obtener los nombres de las imágenes sin la ruta ni la extensión
            sorted_image_names = [get_image_name(img) for img in sorted_images]

            # Crear un ID único para cada usuario
            user_id = len(pd.read_csv(DATA_FILE)) + 1 if os.path.exists(DATA_FILE) else 1  # ID incremental
            df = pd.DataFrame([[user_id] + sorted_image_names], columns=["ID"] + [f"Rank_{i}" for i in range(1, len(sorted_image_names) + 1)])

            # Guardar en un archivo CSV localmente
            local_file = "responses_temp.csv"
            df.to_csv(local_file, index=False)

            # Obtener el token de acceso de Dropbox desde los secretos de Streamlit
            dropbox_token = st.secrets["dropbox"]["access_token"]
            
            # Subir el archivo CSV a Dropbox
            upload_to_dropbox(local_file, DATA_FILE, dropbox_token)

            st.success("Respuesta guardada correctamente. Muchas gracias por tu participación.")

            # Marcar que la respuesta ha sido guardada y ocultar las imágenes
            st.session_state.response_saved = True
            st.session_state.selected_images = []  # Eliminar las imágenes para evitar que se muestren de nuevo
else:
    st.write("¡Ya has respondido la encuesta! Muchas gracias por participar.")
