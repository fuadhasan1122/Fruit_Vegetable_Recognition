import streamlit as st
from PIL import Image
import numpy as np
import os
import requests
from bs4 import BeautifulSoup

# ✅ Always use tensorflow.keras (NOT keras)
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# ✅ Load model safely
@st.cache_resource
def load_my_model():
    return load_model('FV.h5', compile=False)

model = load_my_model()

# ✅ Labels
labels = {
    0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage',
    5: 'capsicum', 6: 'carrot', 7: 'cauliflower', 8: 'chilli pepper',
    9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
    14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
    19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear',
    24: 'peas', 25: 'pineapple', 26: 'pomegranate', 27: 'potato',
    28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
    32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'
}

fruits = [
    'Apple', 'Banana', 'Bell pepper', 'Chilli pepper', 'Grapes',
    'Jalepeno', 'Kiwi', 'Lemon', 'Mango', 'Orange', 'Paprika',
    'Pear', 'Pineapple', 'Pomegranate', 'Watermelon'
]

vegetables = [
    'Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower',
    'Corn', 'Cucumber', 'Eggplant', 'Ginger', 'Lettuce',
    'Onion', 'Peas', 'Potato', 'Raddish', 'Soy beans',
    'Spinach', 'Sweetcorn', 'Sweetpotato', 'Tomato', 'Turnip'
]

# ✅ Safe calorie fetch (fallback if blocked)
def fetch_calories(prediction):
    calorie_data = {
        "Apple": "52 kcal",
        "Banana": "89 kcal",
        "Beetroot": "43 kcal",
        "Bell pepper": "20 kcal",
        "Cabbage": "25 kcal",
        "Capsicum": "20 kcal",
        "Carrot": "41 kcal",
        "Cauliflower": "25 kcal",
        "Chilli pepper": "40 kcal",
        "Corn": "96 kcal",
        "Cucumber": "16 kcal",
        "Eggplant": "25 kcal",
        "Garlic": "149 kcal",
        "Ginger": "80 kcal",
        "Grapes": "69 kcal",
        "Jalepeno": "29 kcal",
        "Kiwi": "61 kcal",
        "Lemon": "29 kcal",
        "Lettuce": "15 kcal",
        "Mango": "60 kcal",
        "Onion": "40 kcal",
        "Orange": "47 kcal",
        "Paprika": "282 kcal",
        "Pear": "57 kcal",
        "Peas": "81 kcal",
        "Pineapple": "50 kcal",
        "Pomegranate": "83 kcal",
        "Potato": "77 kcal",
        "Raddish": "16 kcal",
        "Soy beans": "173 kcal",
        "Spinach": "23 kcal",
        "Sweetcorn": "86 kcal",
        "Sweetpotato": "86 kcal",
        "Tomato": "18 kcal",
        "Turnip": "28 kcal",
        "Watermelon": "30 kcal"
    }

    return calorie_data.get(prediction, "Not available")
# ✅ Image processing
def process_image(img_path):
    img = load_img(img_path, target_size=(224, 224))  # ❗ FIXED (no 3 channel here)
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    y_class = np.argmax(prediction, axis=1)[0]

    result = labels[y_class]
    return result.capitalize()

# ✅ Streamlit App
def run():
    st.title("🍍 Fruits & Vegetable Classifier & Calories Measure 🍅")

    # Ensure folder exists
    if not os.path.exists("upload_images"):
        os.makedirs("upload_images")

    img_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if img_file is not None:
        img = Image.open(img_file)
        st.image(img, width=250)

        save_path = os.path.join("upload_images", img_file.name)
        with open(save_path, "wb") as f:
            f.write(img_file.getbuffer())

        result = process_image(save_path)

        if result in vegetables:
            st.info("Category: Vegetable 🥦")
        else:
            st.info("Category: Fruit 🍎")

        st.success(f"Predicted: {result}")

        calories = fetch_calories(result)
        st.warning(f"Calories: {calories} (per 100g)")

# ✅ Run app
if __name__ == "__main__":
    run()