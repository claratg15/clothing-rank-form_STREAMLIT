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
st.markdown("<h4 style='color: blue;'>Si has obert l'enquesta amb el telèfon mòbil, desactiva la rotació i gira el mòbil en horitzontal per a veure bé les imatges que trobaràs més endavant.</h3>", unsafe_allow_html=True)
#st.write("**Si has obert l'enquesta amb el telèfon mòbil, desactiva la rotació i gira el mòbil en horitzontal per a veure bé les imatges que trobaràs més endavant.**")

st.subheader("**Primera part: Preguntes demogràfiques**")
genere = st.selectbox("Gènere", ["Home", "Dona", "Altres"])
edat = st.number_input("Edat", min_value=1, max_value=100, step=1)
compra_mode = st.selectbox("Com prefereixes comprar articles de roba: de manera online o físicament en botiga?", ["Online", "Físicament en botiga", "Ambdues opcions per igual"])
#compra_mode = st.radio("Com prefereixes comprar articles de roba: de manera online o físicament en botiga?", ["Online", "Físicament en botiga", "Ambdues opcions per igual"])

# Ordenar imatges
st.subheader("**Segona part: Rànquing d'articles de roba**")
st.write("En aquesta secció has d'ordenar les peces de roba que tens a sota segons la teva preferència.")
st.write("Per ordenar les imatges, arrossega les etiquetes de les imatges (de color vermell) a la posició corresponent, movent-les amunt i avall. Les fotografies es reordenen automàticament.")
#"Arrossega les etiquetes de les imatges per ordenar-les segons la teva preferència. Les fotografies s'aniran reordenant segons l'ordre en què les hagis classificat.")

# Creem etiquetes ("Imagen 1", "Imagen 2"...)
image_labels = [f"Imatge {i+1}" for i in range(len(image_list))]

# Ordenem les imatges (mostrant només els noms als elements ordenables)
# sorted_filenames = sort_items(image_labels, direction="vertical")

# !! nou
# Crear la disposició amb dues columnes: una per al número i una per a l'etiqueta ordenable
col1, col2, col3, col4 = st.columns([0.3, 0.15, 0.45, 0.2])  # La primera columna és més estreta per als números

with col2:
    st.write("<u>Rànquing</u>", unsafe_allow_html=True)
    for i in range(len(image_labels)):
        st.markdown(f"<p style='text-align: left;'>Posició {i+1}:</p>", 
                    unsafe_allow_html=True)
#        st.write(f"**Posició {i+1}**")  # Els números es mostren fixes

with col3:
    st.write("<u>Ordena les etiquetes de les imatges:</u>", unsafe_allow_html=True)
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




# --- Append this to the end of your existing Streamlit code ---

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape
from tensorflow.keras.models import Model
from PIL import Image
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import tensorflow as tf

# Set random seeds for reproducibility
np.random.seed(123)
tf.random.set_seed(123)

# --- Autoencoder for Image Feature Extraction ---
def load_and_preprocess_image(image_path, target_size=(64, 64)):
    img = Image.open(image_path).resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    return img_array

# Load all images
image_paths = [os.path.join("subset_100_images", f) for f in os.listdir("subset_100_images") if f.endswith(('.png', '.jpg', '.jpeg'))]
image_data = np.array([load_and_preprocess_image(p) for p in image_paths])

# Define autoencoder
input_img = Input(shape=(64, 64, 3))

# Encoder
encoded = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
encoded = MaxPooling2D((2, 2))(encoded)
encoded = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
encoded = MaxPooling2D((2, 2))(encoded)
encoded = Flatten()(encoded)
encoded = Dense(128, activation='relu')(encoded)  # Latent space

# Decoder
decoded = Dense(16 * 16 * 16)(encoded)
decoded = Reshape((16, 16, 16))(decoded)
decoded = Conv2D(16, (3, 3), activation='relu', padding='same')(decoded)
decoded = UpSampling2D((2, 2))(decoded)
decoded = Conv2D(32, (3, 3), activation='relu', padding='same')(decoded)
decoded = UpSampling2D((2, 2))(decoded)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(decoded)

# Autoencoder and encoder models
autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)

# Compile autoencoder (assuming pre-trained for simplicity)
autoencoder.compile(optimizer='adam', loss='mse')
# Uncomment to train (if needed):
# autoencoder.fit(image_data, image_data, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Extract image features
image_features = encoder.predict(image_data)

# --- Cosine Similarity ---
def cosine_similarity_matrix(feature_matrix):
    norm = np.sqrt(np.sum(feature_matrix ** 2, axis=1))
    normalized = feature_matrix / norm[:, np.newaxis]
    sim_matrix = normalized @ normalized.T
    return sim_matrix

cos_sim_matrix = cosine_similarity_matrix(image_features)
image_names = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]
cos_sim_matrix = pd.DataFrame(cos_sim_matrix, index=image_names, columns=image_names)

# --- Load and Prepare Data ---
# Load existing responses (respostes.csv)
responses = pd.read_csv("responses_temp.csv")

# Assign user ID to each row (1-based indexing)
responses['usuari'] = range(1, len(responses) + 1)

# Set usuari_escollit as the last user (from new_data)
usuari_escollit = responses['usuari'].max()  # Last row corresponds to new_data

# Prepare demographic columns
responses['Home'] = (responses['Gènere'] == 'Home').astype(int)
responses['Dona'] = (responses['Gènere'] == 'Dona').astype(int)
responses['Altres'] = (responses['Gènere'] == 'Altres').astype(int)
responses['Físicament en botiga'] = (responses['Compra'] == 'Físicament en botiga').astype(int)
responses['Online'] = (responses['Compra'] == 'Online').astype(int)
responses['Ambdues opcions'] = (responses['Compra'] == 'Ambdues opcions per igual').astype(int)

# Rename ranking columns to image names without extensions
rank_cols = [f"Rank_{i}" for i in range(1, 11)]
responses[rank_cols] = responses[rank_cols].apply(lambda x: [os.path.splitext(os.path.basename(img))[0] for img in x])

# Tidy ratings data
valoracions = responses[['usuari', 'Home', 'Dona', 'Altres', 'Edat', 'Físicament en botiga', 'Online', 'Ambdues opcions'] + rank_cols]
valoracions_tidy = valoracions.melt(
    id_vars=['usuari', 'Home', 'Dona', 'Altres', 'Edat', 'Físicament en botiga', 'Online', 'Ambdues opcions'],
    value_vars=rank_cols,
    var_name='rank_col',
    value_name='imatge'
)

# Assign ranking values (1 to 10 based on position)
valoracions_tidy['rànquing'] = valoracions_tidy['rank_col'].str.extract(r'(\d+)').astype(int)

# --- Normalize Data ---
scaler = StandardScaler()
valoracions_tidy['rànquing'] = valoracions_tidy.groupby('usuari')['rànquing'].transform(lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten())
valoracions_tidy['Edat'] = scaler.fit_transform(valoracions_tidy[['Edat']])

# Load atributs2 and normalize Edat_model (if available)
try:
    atributs2 = pd.read_csv("atributs2.csv")
    atributs2['Edat_model'] = scaler.fit_transform(atributs2[['Edat_model']])
except FileNotFoundError:
    st.warning("atributs2.csv not found. Skipping Edat_model normalization.")

# --- User-Based Collaborative Filtering ---
# User-item matrix
valoracions_usuaris = valoracions_tidy.pivot(index='imatge', columns='usuari', values='rànquing').fillna(np.nan)
demografics = valoracions_tidy[['usuari', 'Edat', 'Home', 'Dona', 'Altres', 'Físicament en botiga', 'Online', 'Ambdues opcions']].drop_duplicates()

# Similarity function
def correlacio_func_combined(usuari_x, usuari_y, ranquings_data, demo_data):
    sim_finals = pd.DataFrame({'usuari': usuari_x, 'similitud': np.nan})
    
    for t, ux in enumerate(usuari_x):
        # Rankings similarity
        ranquings_x = ranquings_data[ux]
        ranquings_y = ranquings_data[usuari_y]
        sim_ranquings = ranquings_x.corr(ranquings_y, method='pearson')
        if pd.isna(sim_ranquings):
            sim_ranquings = 0
        
        # Demographic similarity
        demo_x = demo_data[demo_data['usuari'] == ux]
        demo_y = demo_data[demo_data['usuari'] == usuari_y]
        
        sim_edat = 1 - abs(demo_x['Edat'].values[0] - demo_y['Edat'].values[0])
        sim_home = int(demo_x['Home'].values[0] == demo_y['Home'].values[0])
        sim_dona = int(demo_x['Dona'].values[0] == demo_y['Dona'].values[0])
        sim_altres = int(demo_x['Altres'].values[0] == demo_y['Altres'].values[0])
        sim_fisicament = int(demo_x['Físicament en botiga'].values[0] == demo_y['Físicament en botiga'].values[0])
        sim_online = int(demo_x['Online'].values[0] == demo_y['Online'].values[0])
        sim_ambdues = int(demo_x['Ambdues opcions'].values[0] == demo_y['Ambdues opcions'].values[0])
        
        demo_sims = [sim_edat, sim_home, sim_dona, sim_altres, sim_fisicament, sim_online, sim_ambdues]
        sim_demo = np.mean([x for x in demo_sims if not np.isnan(x)])
        
        # Combine similarities
        pes_rkgs = 0.6
        pes_demo = 0.4
        sim_total = (pes_rkgs * sim_ranquings) + (pes_demo * sim_demo)
        sim_finals.iloc[t, 1] = sim_total
    
    return sim_finals

resta_usuaris = [u for u in valoracions_usuaris.columns if u != usuari_escollit]
similitud_usuaris = correlacio_func_combined(resta_usuaris, usuari_escollit, valoracions_usuaris, demografics)

# --- Identify Unseen Items ---
# Get all possible images
all_images = set(image_names)
# Get images rated by the user
rated_images = set(valoracions_tidy[valoracions_tidy['usuari'] == usuari_escollit]['imatge'])
# Unseen images
roba_no_vista = list(all_images - rated_images)

# --- Cross-Validation for Optimal k ---
def predict_for_test(train, test, similitud_usuaris, k):
    predictions = []
    for _, row in test.iterrows():
        u, img = row['usuari'], row['imatge']
        usuaris_imatge_i = train[train['imatge'] == img]['usuari'].unique()
        
        if len(usuaris_imatge_i) < 5:
            continue
        
        top_usuaris = similitud_usuaris[similitud_usuaris['similitud'] >= 0][similitud_usuaris['usuari'].isin(usuaris_imatge_i)]
        top_usuaris = top_usuaris.sort_values('similitud', ascending=False).head(k)
        
        if len(top_usuaris) < 3:
            continue
        
        valoracions_top = train[train['imatge'] == img][train['usuari'].isin(top_usuaris['usuari']black)
        top_usuaris = top_usuaris.merge(valoracions_top[['usuari', 'rànquing']], on='usuari')
        
        pred = np.sum(top_usuaris['similitud'] * top_usuaris['rànquing']) / np.sum(top_usuaris['similitud'])
        real = test[(test['usuari'] == u) & (test['imatge'] == img)]['rànquing'].values[0]
        
        if pd.isna(pred) or pd.isna(real):
            continue
        
        predictions.append({'real': real, 'pred': pred})
    
    return pd.DataFrame(predictions)

def cross_validate_k_user(valoracions_tidy_norm, similitud_usuaris, usuari_objectiu, k_values=range(1, 16), folds=5):
    user_data = valoracions_tidy_norm[valoracions_tidy_norm['usuari'] == usuari_objectiu]
    user_data = user_data.copy()  # Avoid SettingWithCopyWarning
    user_data['fold'] = np.random.choice(range(1, folds + 1), size=len(user_data), replace=True)
    
    all_train_data = valoracions_tidy_norm[valoracions_tidy_norm['usuari'] != usuari_objectiu]
    
    results = []
    for k in k_values:
        fold_rmse = []
        for f in range(1, folds + 1):
            test = user_data[user_data['fold'] == f]
            train_ratings = user_data[user_data['fold'] != f]
            train = pd.concat([all_train_data, train_ratings])
            
            preds = predict_for_test(train, test, similitud_usuaris, k)
            if len(preds) == 0:
                fold_rmse.append(np.nan)
                continue
            rmse_val = mean_squared_error(preds['real'], preds['pred'], squared=False)
            fold_rmse.append(rmse_val)
        
        results.append({'k': k, 'rmse': np.nanmean(fold_rmse)})
    
    return pd.DataFrame(results)

# Apply cross-validation
resultats_k_user = cross_validate_k_user(valoracions_tidy, similitud_usuaris, usuari_escollit)
best_k = resultats_k_user[resultats_k_user['rmse'] == resultats_k_user['rmse'].min()]['k'].iloc[0]

# --- Predict Rankings for Unseen Items ---
prediccio_rkg = []
imatge = []
n_obs_prediccio = []

for img in roba_no_vista:
    usuaris_imatge_i = valoracions_tidy[valoracions_tidy['imatge'] == img]['usuari'].unique()
    
    if len(usuaris_imatge_i) < 5:
        continue
    
    top_usuaris = similitud_usuaris[similitud_usuaris['similitud'] >= 0][similitud_usuaris['usuari'].isin(usuaris_imatge_i)]
    top_usuaris = top_usuaris.sort_values('similitud', ascending=False).head(best_k)
    
    if len(top_usuaris) < 3:
        continue
    
    valoracions_top = valoracions_tidy[valoracions_tidy['imatge'] == img][valoracions_tidy['usuari'].isin(top_usuaris['usuari'])]
    top_usuaris = top_usuaris.merge(valoracions_top[['usuari', 'rànquing']], on='usuari')
    
    pred = np.sum(top_usuaris['similitud'] * top_usuaris['rànquing']) / np.sum(top_usuaris['similitud'])
    
    prediccio_rkg.append(pred)
    imatge.append(img)
    n_obs_prediccio.append(len(top_usuaris))

# --- Top-10 Recommendations ---
top10_recomanacions_ub = pd.DataFrame({
    'imatge': imatge,
    'prediccio_rkg': prediccio_rkg,
    'n_obs_prediccio': n_obs_prediccio
}).sort_values('prediccio_rkg').head(10)

# --- Display Top-3 Recommendations and Allow Rating ---
st.subheader("Les teves recomanacions personalitzades")
st.write("A continuació es mostren les 3 peces de roba recomanades per a tu. Si us plau, puntua cada imatge de l'1 al 10.")

top3_recomanacions = top10_recomanacions_ub.head(3)
ratings = {}

for idx, row in top3_recomanacions.iterrows():
    img_name = row['imatge']
    img_path = os.path.join("subset_100_images", f"{img_name}.jpg")  # Adjust extension if needed
    st.image(img_path, caption=f"Peça de roba: {img_name}", use_container_width=True)
    rating = st.number_input(f"Puntua aquesta peça ({img_name}) de l'1 al 10", min_value=1, max_value=10, step=1, key=f"rating_{img_name}")
    ratings[img_name] = rating

if st.button("Enviar puntuacions"):
    st.write("Puntuacions rebudes:")
    for img_name, rating in ratings.items():
        st.write(f"Peça {img_name}: {rating}/10")
    st.success("Gràcies per les teves puntuacions!")
    # Optionally save ratings to a file or Dropbox