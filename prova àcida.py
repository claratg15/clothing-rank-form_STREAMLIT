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


if "ranking_submitted" not in st.session_state:
    st.session_state.ranking_submitted = False
if "ratings_submitted" not in st.session_state:
    st.session_state.ratings_submitted = False
if "ratings" not in st.session_state:
    st.session_state.ratings = {}
if "top3_recomanacions" not in st.session_state:
    st.session_state.top3_recomanacions = None
if "sorted_images" not in st.session_state:
    st.session_state.sorted_images = None


if st.button("Ja tinc el meu rànquing final"):
    if not st.session_state.ranking_submitted:
        st.write("Si us plau, espera mentres processem la teva resposta. Moltes gràcies per la teva participació!")
        sorted_image_names = [get_image_name(img) for img in sorted_images]
        new_data = pd.DataFrame([[genere, edat, compra_mode] + sorted_image_names], 
        columns=["Gènere", "Edat", "Compra"] + [f"Rank_{i}" for i in range(1, len(sorted_image_names) + 1)])

        st.session_state.ranking_submitted = True
        st.session_state.sorted_images = sorted_images

        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        #from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape
        #from tensorflow.keras.models import Model
        from PIL import Image
        import os
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.metrics import mean_squared_error
        #import tensorflow as tf
        import streamlit as st
        import os
        import random
        import pandas as pd
        import dropbox
        import requests
        from streamlit_sortables import sort_items


        # Configuració de la pàgina

        # Carpeta on estan les fotos
        IMAGE_FOLDER = "subset_100_images" 

        np.random.seed(123)
        image_paths = [os.path.join("subset_100_images", f) for f in os.listdir("subset_100_images") if f.endswith(('.png', '.jpg', '.jpeg'))]
        image_names = [os.path.basename(p) for p in image_paths]


        # Carreguem les dades
        try:
            valoracions = pd.read_csv("valoracions.csv")
        except FileNotFoundError:
            st.error("valoracions.csv not found. Please ensure the file exists.")
            st.stop()

        # Preparem les dades de la nova observació
        genere = new_data.iloc[0]['Gènere']
        compra = new_data.iloc[0]['Compra']
        edat = new_data.iloc[0]['Edat']

        # Creem nova fila
        new_row = {
            'usuari': valoracions['usuari'].max() + 1,
            'Home': 1 if genere == 'Home' else 0,
            'Dona': 1 if genere == 'Dona' else 0,
            'Altres': 1 if genere == 'Altres' else 0,
            'Edat': edat,
            'Físicament en botiga': 1 if compra == 'Físicament en botiga' else 0,
            'Online': 1 if compra == 'Online' else 0,
            'Ambdues opcions': 1 if compra == 'Ambdues opcions per igual' else 0
        }

        # Inicialitzem les columnes d’imatges amb NaN
        for col in valoracions.columns:
            if col not in new_row:
                new_row[col] = np.nan

        # Convertim la nova fila en DataFrame i l'afegim a valoracions
        new_row_df = pd.DataFrame([new_row])
        valoracions = pd.concat([valoracions, new_row_df], ignore_index=True)

        # Assignem valors de rànquing a la nova fila
        # nova fila està a l’última posició
        new_index = len(valoracions) - 1

        # Recorrem les columnes del rànquing (rank_1, rank_2, etc.)
        for j in range(3, new_data.shape[1]):  # des de la 4a columna en R
            image_name = str(new_data.iat[0, j]) + ".jpg"  # valor de la cel·la (nom de la imatge)
            if image_name in valoracions.columns:
                try:
                    rank_str = new_data.columns[j][5:7]  # extreu "1", "2", ..., "10"
                    valoracions.at[new_index, image_name] = int(rank_str)
                except ValueError:
                    valoracions.at[new_index, image_name] = np.nan


        # Definim usuari_escollit com a l'últim usuari
        usuari_escollit = 109
        valoracions_tidy = valoracions.melt(
            id_vars=['usuari', 'Home', 'Dona', 'Altres', 'Edat', 'Físicament en botiga', 'Online', 'Ambdues opcions'],
            var_name='imatge',
            value_name='rànquing'
        )


        # Normalitzem les dades
        scaler = StandardScaler()
        valoracions_tidy['rànquing'] = valoracions_tidy.groupby('usuari')['rànquing'].transform(lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten())
        valoracions_tidy['Edat'] = scaler.fit_transform(valoracions_tidy[['Edat']])



        # USER-BASED
        # Matriu usuari-ítem
        valoracions_usuaris = valoracions_tidy.pivot(index='imatge', columns='usuari', values='rànquing').fillna(np.nan)
        demografics = valoracions_tidy[['usuari', 'Edat', 'Home', 'Dona', 'Altres', 'Físicament en botiga', 'Online', 'Ambdues opcions']].drop_duplicates()
        # print(valoracions_usuaris)

        # Funció per calcular la similitud
        def correlacio_func_combined(usuari_x, usuari_y, ranquings_data, demo_data):
            sim_finals = pd.DataFrame({'usuari': usuari_x, 'similitud': np.nan})
            
            for t, ux in enumerate(usuari_x):
                # Similitud per rànquings
                ranquings_x = ranquings_data[ux]
                ranquings_y = ranquings_data[usuari_y]
                sim_ranquings = ranquings_x.corr(ranquings_y, method='pearson')
                if pd.isna(sim_ranquings):
                    sim_ranquings = 0
                
                # Similitud demogràfica
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
                
                # Similituds combinades
                pes_rkgs = 0.6
                pes_demo = 0.4
                sim_total = (pes_rkgs * sim_ranquings) + (pes_demo * sim_demo)
                sim_finals.iloc[t, 1] = sim_total
            
            return sim_finals

        resta_usuaris = [u for u in valoracions_usuaris.columns if u != usuari_escollit]
        similitud_usuaris = correlacio_func_combined(resta_usuaris, usuari_escollit, valoracions_usuaris, demografics)
        #print(similitud_usuaris)

        # Imatges no vistes
        all_images = set(image_names)
        roba_no_vista = valoracions_tidy[(valoracions_tidy['usuari'] == (usuari_escollit)) & (valoracions_tidy['rànquing'].isna())]['imatge'].unique().tolist()

        # Cross-validació per la millor k
        def predict_for_test(train, test, similitud_usuaris, k):
            predictions = []

            for _, row in test.iterrows():
                u, img = row['usuari'], row['imatge']
                usuaris_imatge_i = train[train['imatge'] == img][train['rànquing'].notna()]['usuari'].unique()
                
                if len(usuaris_imatge_i) < 5:
                    continue
                
                top_usuaris = similitud_usuaris[similitud_usuaris['similitud'] >= 0][similitud_usuaris['usuari'].isin(usuaris_imatge_i)]
                top_usuaris = top_usuaris.sort_values('similitud', ascending=False).head(k)
                
                if len(top_usuaris) < 3:
                    continue
                
                valoracions_top = train[train['imatge'] == img][train['usuari'].isin(top_usuaris['usuari'])]
                top_usuaris = top_usuaris.merge(valoracions_top[['usuari', 'rànquing']], on='usuari', how='left')
                top_usuaris = top_usuaris.sort_values('similitud', ascending=False).head(k)
                
                pred = np.sum(top_usuaris['similitud'] * top_usuaris['rànquing'].fillna(0)) / np.sum(top_usuaris['similitud'])

                real = test[(test['usuari'] == u) & (test['imatge'] == img)]['rànquing'].values
                
                if pd.isna(pred) or pd.isna(real):
                    print(f"Imatge {img} saltada: pred o real és NaN (pred={pred}, real={real})")
                    continue
                
                predictions.append({'real': real[0], 'pred': pred})

            print(f"Prediccions generades: {len(predictions)}")
            return pd.DataFrame(predictions)


        def cross_validate_k_user(valoracions_tidy_norm, similitud_usuaris, usuari_objectiu, k_values=range(1, 16), folds=5):
            user_data = valoracions_tidy_norm[valoracions_tidy_norm['usuari'] == (usuari_objectiu)][valoracions_tidy_norm['rànquing'].notna()].copy()


            if user_data.empty:
                print(f"Error: No hi ha dades per a l'usuari {usuari_objectiu}")
                return pd.DataFrame()
            
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
                    mse_val = mean_squared_error(preds['real'], preds['pred'])
                    rmse_val = np.sqrt(mse_val)
                    fold_rmse.append(rmse_val)
                
                results.append({'k': k, 'rmse': np.nanmean(fold_rmse)})
            return pd.DataFrame(results)

        resultats_k_user = cross_validate_k_user(valoracions_tidy, similitud_usuaris, usuari_escollit)
        print(resultats_k_user)

        best_k = resultats_k_user[resultats_k_user['rmse'] == resultats_k_user['rmse'].min()]['k'].iloc[0]
        print(best_k)

        # Predim els rànquings per les imatges no vistes
        prediccio_rkg = np.array([np.nan] * len(roba_no_vista))
        imatge = np.array([np.nan] * len(roba_no_vista), dtype=object)
        n_obs_prediccio = np.array([np.nan] * len(roba_no_vista))


        for i in range(len(roba_no_vista)):
            if roba_no_vista[i] not in valoracions_tidy['imatge'].values:
                print(f"Imatge {roba_no_vista[i]} (tipus: {type(roba_no_vista[i])}) no trobada a valoracions_tidy")
                continue

            valoracions_tidy['imatge'] = valoracions_tidy['imatge'].astype(str)
            roba_no_vista = [str(img) for img in roba_no_vista]
            usuaris_imatge_i = valoracions_tidy[(valoracions_tidy['imatge'] == roba_no_vista[i]) & (valoracions_tidy['rànquing'].notna())]['usuari'].unique()

            if len(usuaris_imatge_i) < 5:
                continue
            
            top_usuaris = similitud_usuaris[(similitud_usuaris['similitud'] >= 0) & (similitud_usuaris['usuari'].isin(usuaris_imatge_i))]
            top_usuaris = top_usuaris.sort_values('similitud', ascending=False).head(best_k)
            
            if len(top_usuaris) < 3:
                continue
            
            valoracions_top = valoracions_tidy[valoracions_tidy['imatge'] == roba_no_vista[i]][valoracions_tidy['usuari'].isin(top_usuaris['usuari'])]
            valoracions_top['usuari'] = valoracions_top['usuari'].astype(str)
            top_usuaris['usuari'] = top_usuaris['usuari'].astype(str)

            top_usuaris = top_usuaris.merge(valoracions_top[['usuari', 'rànquing']], on='usuari', how='left')

            prediccio_rkg[i] = np.sum(top_usuaris['similitud'] * top_usuaris['rànquing']) / np.sum(top_usuaris['similitud'])    
            imatge[i] = roba_no_vista[i]
            n_obs_prediccio[i] = len(top_usuaris)


        # Top-10 recomanacions
        top10_recomanacions_ub = pd.DataFrame({
            'imatge': imatge,
            'prediccio_rkg': prediccio_rkg,
            'n_obs_prediccio': n_obs_prediccio
        }).sort_values('prediccio_rkg').head(10)

        st.session_state.top3_recomanacions = top10_recomanacions_ub.head(3)

    # Mostrem les recomanacions per a que les puntui
if st.session_state.ranking_submitted and not st.session_state.ratings_submitted:
    # Nom de l'arxiu de Dropbox on guardarem les respostes
    DATA_FILE = "/respostes_prova_acida.csv"
    if st.session_state.top3_recomanacions is None or st.session_state.top3_recomanacions.empty:
        st.warning("No hi ha recomanacions disponibles. Revisa les dades o el procés de predicció.")
    else:
        st.subheader("Les teves recomanacions personalitzades")
        st.write("A continuació es mostren les 3 peces de roba recomanades per a tu. Si us plau, puntua cada imatge de l'1 al 10.")
        
        cols = st.columns(3)
        
        for idx, (col, row) in enumerate(zip(cols, st.session_state.top3_recomanacions.iterrows())):
            img_name = row[1]['imatge']
            img_path = os.path.join("subset_100_images", f"{img_name}")

            with col:
                if img_path and os.path.exists(img_path):
                    st.image(img_path, use_container_width=True)
                    rating = st.slider(
                        f"Puntuació per {img_name}:",
                        min_value=1, max_value=10, step=1,
                        key=f"rating_{img_name}",
                        value=st.session_state.ratings.get(img_name, 1)
                    )
                    st.session_state.ratings[img_name] = rating
                else:
                    st.error(f"No s'ha trobat la imatge per a {img_name}. Comprova el nom o l'extensió.")

        # Per enviar les puntuacions
        if st.button("Enviar puntuacions"):
            if st.session_state.ratings:
                sorted_image_names = [get_image_name(img) for img in st.session_state.sorted_images]
                recomanacions = list(st.session_state.top3_recomanacions['imatge'])
                puntuacions = [st.session_state.ratings.get(img, None) for img in recomanacions]
                new_data2 = pd.DataFrame(
                    [[genere, edat, compra_mode, *sorted_image_names, *recomanacions, *puntuacions]],
                    columns=[
                        "Gènere", "Edat", "Compra",
                        *[f"Rank_{i}" for i in range(1, len(sorted_image_names) + 1)],
                        "Recom_1", "Recom_2", "Recom_3",
                        "Rating_1", "Rating_2", "Rating_3"
                    ]
                )

                # Guardem totes les dades
                local_file = "responses_temp.csv"
                download_from_dropbox(DATA_FILE, local_file)
                if os.path.exists(local_file):
                    df = pd.read_csv(local_file)
                    df = pd.concat([df, new_data2], ignore_index=True)
                else:
                    df = new_data2
                df.to_csv(local_file, index=False)
                upload_to_dropbox(local_file, DATA_FILE)

                # Per mostrar resum de les puntuacions que ha fet
                st.write("Puntuacions rebudes:")
                for img_name, rating in st.session_state.ratings.items():
                    st.write(f"Peça {img_name}: {rating}/10")
                st.success("Gràcies per les teves puntuacions!")

                st.session_state.ratings_submitted = True
else:
    if st.session_state.ratings_submitted:
        st.write("Ja has respost l'enquesta. Moltes gràcies per la teva participació!")
    elif not st.session_state.ranking_submitted:
        st.write("Si us plau, completa el rànquing d'imatges abans de continuar.")
