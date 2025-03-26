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
st.markdown("<h3 style='color: blue;'>i has obert l'enquesta amb el telèfon mòbil, desactiva la rotació i gira el mòbil en horitzontal per a veure bé les imatges que trobaràs més endavant.</h3>", unsafe_allow_html=True)
#st.write("**Si has obert l'enquesta amb el telèfon mòbil, desactiva la rotació i gira el mòbil en horitzontal per a veure bé les imatges que trobaràs més endavant.**")

st.subheader("**Primera part: Preguntes demogràfiques")
genere = st.selectbox("Gènere", ["Home", "Dona", "Altres"])
edat = st.number_input("Edat", min_value=1, max_value=100, step=1)
compra_mode = st.selectbox("Com prefereixes comprar articles de roba: de manera online o físicament en botiga?", ["Online", "Físicament en botiga", "Ambdues opcions per igual"])
#compra_mode = st.radio("Com prefereixes comprar articles de roba: de manera online o físicament en botiga?", ["Online", "Físicament en botiga", "Ambdues opcions per igual"])

# Ordenar imatges
st.subheader("**Segona part: Rànquing d'articles de roba")
st.write("**A sota tens una columna que indica la posició del rànquing, i una columna amb les etiquetes de les imatges en vermell. Les fotografies es troben a sota d'aquestes dues columnes. "
"**Arrossega les etiquetes de les imatges per ordenar-les segons la teva preferència. Les fotografies s'aniran reordenant segons l'ordre en què les hagis classificat.**")

# Creem etiquetes ("Imagen 1", "Imagen 2"...)
image_labels = [f"Imatge {i+1}" for i in range(len(image_list))]

# Ordenem les imatges (mostrant només els noms als elements ordenables)
# sorted_filenames = sort_items(image_labels, direction="vertical")

# !! nou
# Crear la disposició amb dues columnes: una per al número i una per a l'etiqueta ordenable
col1, col2, col3, col4 = st.columns([0.3, 0.1, 0.4, 0.5])  # La primera columna és més estreta per als números

with col2:
    st.write("<u>Rànquing</u>", unsafe_allow_html=True)
    for i in range(len(image_labels)):
        st.markdown(f"<p style='text-align: left;'>Posició {i+1}:</p>", 
                    unsafe_allow_html=True)
#        st.write(f"**Posició {i+1}**")  # Els números es mostren fixes

with col3:
    st.write("<u>Ordena les imatges segons la teva preferència:</u>", unsafe_allow_html=True)
    st.markdown("""
    <style>
        .stSortableItem {
            padding: 10px;        /* Augmenta l'espai dins de cada etiqueta */
            font-size: 18px;      /* Opcional: augmenta la mida de la font */
        }
    </style>
""", unsafe_allow_html=True)
    sorted_filenames = sort_items(image_labels, direction="vertical")


# Reordenem les imatges en funció de l'ordre de les etiquetes
sorted_images = [os.path.join(IMAGE_FOLDER, image_list[image_labels.index(label)]) for label in sorted_filenames]

# Mostrem les imatges ordenades només si la resposta no ha estat guardada
if not st.session_state.response_saved:
    st.write(" **Imatges segons l'ordre seleccionat:**")
    
    # Mostrar les imatges en columnes (5 fotos per fila)
    cols = st.columns(5)

    # !! nou:
    for i, label in enumerate(sorted_filenames):  # Iterem sobre els noms ordenats
        img_path = os.path.join(IMAGE_FOLDER, image_list[image_labels.index(label)])  # Busquem la imatge corresponent
        
        with cols[i % 5]:  
            st.write(f"**Posició {i+1}:** {label}")  # Mostrem el número fix + etiqueta de la imatge
            st.image(img_path, use_container_width=True)

    #for i, img in enumerate(sorted_images):
    #    with cols[i % 5]:  
    #        st.write(f"**{sorted_filenames[i]}**")  
    #        st.image(img, use_container_width=True)  

# Agafem només el nom de la foto, sense l'extensió .jpg
def get_image_name(image_path):
    return os.path.splitext(os.path.basename(image_path))[0]

# Botó per guardar la resposta
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

            st.success("Resposta enviada correctament. Moltes gràcies per la teva participació!")
            st.session_state.response_saved = True
            st.session_state.selected_images = []  
else:
    st.write("Ja has respost l'enquesta. Moltes gràcies per la teva participació!")
