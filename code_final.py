
import os
import numpy as np
import shutil
import sqlite3
import mysql.connector
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow import keras
from google.cloud import dialogflow_v2 as dialogflow
import streamlit as st

# Configuration Dialogflow
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "private_key.json"
PROJECT_ID = "chatbot-kbgy"
LANGUAGE_CODE = "fr"
SESSION_ID = "session_id"

# Chargement du modèle
with open("model_arch.json", "r") as json_file:
    loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)
model.load_weights("my_model.weights.h5")



# Fonction pour prédire une image
def predict_image(image_path):
    class_names=[
    'Bab Mansour',
    'Cité portugaise',
    'Koutoubia',
    'Kasbah des Oudayas',
    'La Koubba Almoravide à Marrakech',
    'Kasbah Aït Ben Haddou',
    'La mosquée Al Quaraouiyine',
    'Médina de Tétouan',
    'La synagogue Ibn Danan',
    'La tour Hassan',
    'La ville de Chefchaouen',
    'Le palais Bahia à Marrakech',
    'Le palais El Badi à Marrakech',
    'Le palais royal de Fès',
    'Le site archéologique de Lixus',
    'Les grottes d’Hercule à Tanger',
    'Les ruines de Volubilis',
    'Place Jemaa el-Fna',
    'Place Jemaa el-Fna',
    'Le château d’Aïn Asserdoun',
    'Le mausolée Mohammed V',
    'Le mausolée Moulay Idriss Zerhoun',
    'Les portes de Meknès',
    'Mosquée Hassan II'
    ]
    image = keras.utils.load_img(image_path)
    input_arr = keras.utils.img_to_array(image)
    image = np.expand_dims(input_arr, axis=0)
    image = preprocess_input(image)
    predictions = model.predict(image)
    # Find the class with the highest probability
    predicted_class_index = np.argmax(predictions, axis=-1)[0] 
    predicted_class_name = class_names[predicted_class_index]  # Map index to class name
    predicted_probability = predictions[0][predicted_class_index]  # Probability of the top class
    # Print the results
    print(f"Predicted Class: {predicted_class_name}")
    print(f"Probability: {predicted_probability:.2f}")
    #print(predictions.shape)
    #results = decode_predictions(predictions, top=3)
    return [{"nom":predicted_class_name , "probability":predicted_probability } ]
    

# Fonction Dialogflow
def detect_intent_texts(project_id, session_id, text, language_code="fr"):
    session_client = dialogflow.SessionsClient()
    session = session_client.session_path(project_id, session_id)
    text_input = dialogflow.types.TextInput(text=text, language_code=language_code)
    query_input = dialogflow.types.QueryInput(text=text_input)
    response = session_client.detect_intent(session=session, query_input=query_input)
    return response.query_result.fulfillment_text

# Connexion MySQL
def get_id_by_label(label):
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="casa",
            database="monu_hist"
        )
        cursor = conn.cursor()
        query = f"SELECT id FROM monument WHERE nom like '{nom}'"
        print(query)
        cursor.execute(query)
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        return result[0] if result else None
    except mysql.connector.Error as e:
        return None

def info_monument(id):
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="casa",
            database="monu_hist"
        )
        cursor = conn.cursor()
        cursor.execute("SELECT nom,localisation ,description,histoire FROM monument WHERE id = %s", (id,))
        monument_info = cursor.fetchone()
        cursor.close()
        conn.close()
        return monument_info
    except mysql.connector.Error as e:
        return f"Erreur : {e}"

st.set_page_config(page_title="ChatBot Monument" ,page_icon=":monument:")

placeholder = st.empty()
# Interface Streamlit
# st.image("https://imgur.com/dKcJGFq.png", width=200)
# st.set_page_config(page_title="ChatBot Monument" ,page_icon="")
st.title("_Welcome_ to your :green[chatbot] :cloud:...​")
# st.subheader("_Streamlit_ is :blue[chatbot] :sunglasses:")

st.subheader("Posez vos questions ou téléchargez une image pour obtenir des informations  sur le monument ...", divider=True)

# st.subtitle("Posez vos questions ou téléchargez une image pour obtenir des informations  sur le monument ...")

prompt = st.text_input("Posez une question :")
if st.button("Envoyer"):
    if prompt.strip():
        placeholder.text(("patienter svp...✅​✅​ "))
        response = detect_intent_texts(PROJECT_ID, SESSION_ID, prompt, LANGUAGE_CODE)
        st.write("Réponse :", response)
    else:
        st.warning("Veuillez entrer une question.")
        
st.sidebar.image("https://i.postimg.cc/9XG2dGRC/logo.png", width=150)
uploaded_file = st.sidebar.file_uploader("Chargez une image du monument :", type=["jpg", "jpeg", "png"])
if uploaded_file:
    placeholder.text(("patienter svp...✅​✅​ "))
    temp_dir = "temp_images"
    os.makedirs(temp_dir, exist_ok=True)
    img_path = os.path.join(temp_dir, uploaded_file.name)

    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    placeholder.text("Image chargée avec succès ! ✅")
    predictions = predict_image(img_path)

    if predictions:
        nom = predictions[0]["nom"]
        id = get_id_by_label(nom)
        if id:
            monument_info = info_monument(id)
            if monument_info:
                st.write(f"Nom du monument : {monument_info[0]}")
                # st.write(f"localisation du monument : {monument_info[1]}")
                url = monument_info[1]
                st.markdown(f"[Cliquez ici pour voir sur Google Maps]({url})")
                st.write(f"description du monument : {monument_info[2]}")
                st.write(f"histoire du monument : {monument_info[3]}")

                print((monument_info))


            else:
                st.write("Monument non trouvé dans la base de données.")
        else:
            st.write("Monument non identifié.")
    # shutil.rmtree(temp_dir)
    import errno, os, stat, shutil
    def handleRemoveReadonly(func, path, exc):
        excvalue = exc[1]
        if func in (os.rmdir, os.remove) and excvalue.errno == errno.EACCES:
            os.chmod(path, stat.S_IRWXU| stat.S_IRWXG| stat.S_IRWXO) # 0777
            func(path)
        else:
            raise

    shutil.rmtree(temp_dir, ignore_errors=False, onerror=handleRemoveReadonly)


# In[ ]:


#app = FastAPI()

#@app.post("/predict")
#sync def predict(file: UploadFile = File(...)):
    #try:
    #    temp_dir = "temp"
     #   os.makedirs(temp_dir, exist_ok=True)
     #   img_path = os.path.join(temp_dir, file.filename)

      #  with open(img_path, "wb") as f:
       #     shutil.copyfileobj(file.file, f)

       # predictions = predict_image(img_path)
       # label = predictions["predictions"][0]["label"] if predictions["predictions"] else None
       # monument_id = get_id_by_label(label) if label else None

       # os.remove(img_path)

      # if monument_id:
        #    monument_info = info_monument(monument_id)
         #   if monument_info:
          #      return {"monument_name": monument_info[0], "details": monument_info[0]}
          #  else:
           #     return {"error": "Monument not found in the database."}
       # else:
       #     return {"error": "Monument not identified."}
   # except Exception as e:
      #  return JSONResponse(content={"error": str(e)})


# In[1]:





# In[ ]:




