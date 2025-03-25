# FairFace Streamlit App

This is a simple Streamlit app to classify race (Asian, Black, or White) using a deep learning model trained on the FairFace dataset.

## 📦 Files
- `app.py`: Main Streamlit app
- `fairface_race_classifier.h5`: Your trained Keras model (you need to add it manually)

## 🚀 How to Run

1. Move your trained model file (`fairface_race_classifier.h5`) into this folder.
2. Install dependencies:
   ```
   pip install streamlit tensorflow pillow
   ```
3. Run the app:
   ```
   streamlit run app.py
   ```

Enjoy!
