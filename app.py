import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px

# plt = platform.system()
# if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath


st.write("""
# Transport classification model
A model that identifies the type of transport from the image you have uploaded \n
*P.S. Please upload only transport images, others are not supported*         
""")




file = st.file_uploader('Upload image', type=['png', 'jpg', 'jpeg', 'gif', 'svg'])
if file:
    st.image(file)
    img = PILImage.create(file)

    
    model = load_learner('transport_types.pkl')

    pred, pred_id, probs = model.predict(img)
    st.success(f"Prediction: {pred}")
    st.info(f"Probability: {probs[pred_id]*100:.2f}%")

    # plotting
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)
