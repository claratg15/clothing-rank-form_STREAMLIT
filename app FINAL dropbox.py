import streamlit as st
import os
import random
import pandas as pd
import dropbox
import requests
from streamlit_sortables import sort_items

# Configuració de la pàgina
st.set_page_config(layout="wide")

# Obtenir credencials de Dropbox des de secrets.toml
APP_KEY = st.secrets["dropbox"]["app_key"]
APP_SECRET = st.secrets["dropbox"]["app_secret"]
REFRESH_TOKEN = st.secrets["dropbox"]["refresh_token"]

# Funció per generar un access_token a partir del refresh_token
def get_access_token():
    url = "https://api.dropbox.com/oauth2/token"
    data = {
        "grant_type": "refresh_token",
        "refresh_token": REFRESH_TOKEN,
        "client_id": APP_KEY,
        "client_secret": APP_SECRET
    }
    
    response = requests.post(url, data=data)
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        st.error("Error obtenint el token d'accés de Dropbox.")
        return None

# Funció per pujar l'arxiu CSV a Dropbox
def upload_to_dropbox(file_path, dropbox_path):
    access_token = get_access_token()
    if access_token:
        dbx = dropbox.Dropbox(access_token)
        with open(file_path, "rb") as f:
            dbx.files_upload(f.read(), dropbox_path, mode=dropbox.files.WriteMode.overwrite)

# Funció per descarregar l'arxiu CSV des de Dropbox
def download_from_dropbox(dropbox_path, local_path):
    access_token = get_access_token()
    if access_token:
        dbx = dropbox.Dropbox(access_token)
        try:
            metadata, res = dbx.files_download(dropbox_path)
            with open(local_path, "wb") as f:
                f.write(res.content)
        except dropbox.exceptions.ApiError:
            pass  # Si el fitxer no existeix, simplement el crearem més tard

# Carpeta on estan les fotos
IMAGE_FOLDER = "subset_100_images" 
DATA_FILE = "/respostes.csv" 

# Selecció d'imatges
if "selected_images" not in st.session_state:
    def get_random_images(folder, n=10):
        images = [f for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        return random.sample(images, min(n, len(images)))

    st.session_state.selected_images = get_random_images(IMAGE_FOLDER)

image_list = st.session_state.selected_images
if "response_saved" not in st.session_state:
    st.session_state.response_saved = False

# Enquesta
st.title("Enquesta de preferència de peces de roba")
st.write("**Ordena les imatges segons la teva preferència.**")

genere = st.selectbox("Gènere", ["Home", "Dona", "No Binari", "Altres"])
edat = st.number_input("Edat", min_value=10, max_value=100, step=1)
compra_mode = st.radio("Preferència de compra", ["Online", "Físicament en botiga"])

# Ordenar imatges
image_labels = [f"Imagen {i+1}" for i in range(len(image_list))]
sorted_filenames = sort_items(image_labels, direction="vertical")
sorted_images = [os.path.join(IMAGE_FOLDER, image_list[image_labels.index(label)]) for label in sorted_filenames]

if not st.session_state.response_saved:
    cols = st.columns(5)
    for i, img in enumerate(sorted_images):
        with cols[i % 5]:  
            st.image(img, use_container_width=True)  

def get_image_name(image_path):
    return os.path.splitext(os.path.basename(image_path))[0]

if not st.session_state.response_saved:
    if st.button("Enviar resposta"):
        if sorted_images:
            sorted_image_names = [get_image_name(img) for img in sorted_images]
            new_data = pd.DataFrame([[genere, edat, compra_mode] + sorted_image_names], 
                                    columns=["Gènere", "Edat", "Compra"] + [f"Rank_{i}" for i in range(1, len(sorted_image_names) + 1)])

            local_file = "responses_temp.csv"
            download_from_dropbox(DATA_FILE, local_file)

            if os.path.exists(local_file):
                df = pd.read_csv(local_file)
                df = pd.concat([df, new_data], ignore_index=True)
            else:
                df = new_data  

            df.to_csv(local_file, index=False)
            upload_to_dropbox(local_file, DATA_FILE)

            st.success("Resposta enviada correctament!")
            st.session_state.response_saved = True
            st.session_state.selected_images = []  
else:
    st.write("Ja has respost l'enquesta. Moltes gràcies!")
