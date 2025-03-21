import streamlit as st
import os
import random
import pandas as pd
import dropbox
from streamlit_sortables import sort_items

# Configuració de la pàgina
st.set_page_config(layout="wide")

# Carpeta on estan les fotos
IMAGE_FOLDER = "subset_100_images" 

# Nom de l'arxiu de Dropbox on guardarem les respostes
DATA_FILE = "/respostes.csv" 

# Mantenim les mateixes imatges durant la sessió
if "selected_images" not in st.session_state:
    def get_random_images(folder, n=10):
        images = [f for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        return random.sample(images, min(n, len(images)))

    st.session_state.selected_images = get_random_images(IMAGE_FOLDER)

# Llista de les imatges seleccionades
image_list = st.session_state.selected_images

# Marca si la resposta ha estat guardada, per evitar noves respostes
if "response_saved" not in st.session_state:
    st.session_state.response_saved = False

# Títol de l'enquesta
st.title("Enquesta de preferència de peces de roba")
st.write("**Arrossega les etiquetes de les imatges per ordenar-les segons la teva preferència. Les fotografies es troben a sota del requadre.**")

# Creem etiquetes ("Imagen 1", "Imagen 2"...)
image_labels = [f"Imagen {i+1}" for i in range(len(image_list))]

# Ordenem les imatges (mostrant només els noms als elements ordenables)
sorted_filenames = sort_items(image_labels, direction="vertical")

# Reordenem les imatges en funció de l'ordre de les etiquetes
sorted_images = [os.path.join(IMAGE_FOLDER, image_list[image_labels.index(label)]) for label in sorted_filenames]

# Mostrem les imatges ordenades només si la resposta no ha estat guardada
if not st.session_state.response_saved:
    st.write(" **Imatges segons l'ordre seleccionat:**")

    # Mostrar les imatges en columnes (5 fotos per fila)
    cols = st.columns(5)

    for i, img in enumerate(sorted_images):
        with cols[i % 5]:  
            st.write(f"**{sorted_filenames[i]}**")  
            st.image(img, use_container_width=True)  

# Agafem només el nom de la foto, sense l'extensió .jpg
def get_image_name(image_path):
    return os.path.splitext(os.path.basename(image_path))[0]

# Funció per pujar l'arxiu CSV a Dropbox
def upload_to_dropbox(file_path, dropbox_path, token):
    dbx = dropbox.Dropbox(token)
    
    with open(file_path, "rb") as f:
        dbx.files_upload(f.read(), dropbox_path, mode=dropbox.files.WriteMode.overwrite)  

# Funció per descarregar l'arxiu CSV desde Dropbox
def download_from_dropbox(dropbox_path, local_path, token):
    dbx = dropbox.Dropbox(token)
    try:
        metadata, res = dbx.files_download(dropbox_path)
        with open(local_path, "wb") as f:
            f.write(res.content)
    except dropbox.exceptions.ApiError as e:
        
        if isinstance(e.error, dropbox.files.DownloadError):
            pass 


# Botó per guardar la resposta
if not st.session_state.response_saved:
    if st.button("Enviar resposta"):
        if sorted_images:
            
            sorted_image_names = [get_image_name(img) for img in sorted_images]

            
            new_data = pd.DataFrame([sorted_image_names], columns=[f"Rank_{i}" for i in range(1, len(sorted_image_names) + 1)])

            
            dropbox_token = st.secrets["dropbox"]["access_token"]
            
            
            local_file = "responses_temp.csv"
            download_from_dropbox(DATA_FILE, local_file, dropbox_token)

            
            if os.path.exists(local_file):
                df = pd.read_csv(local_file)
                df = pd.concat([df, new_data], ignore_index=True)
            else:
                df = new_data  

            
            df.to_csv(local_file, index=False)

            
            upload_to_dropbox(local_file, DATA_FILE, dropbox_token)

            st.success("Resposta enviada correctament. Moltes gràcies per la teva participació!")

            
            st.session_state.response_saved = True
            st.session_state.selected_images = []  
else:
    st.write("Ja has respost l'enquesta. Moltes gràcies per la teva participació!")
