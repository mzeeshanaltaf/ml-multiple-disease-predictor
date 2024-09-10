import streamlit as st
from streamlit_option_menu import option_menu
from util import *
from models import *

supported_models = ["Logistic Regression", "Support Vector Machines", "Decision Tree", "Random Forest", "Gaussian NB"]

# Initialize streamlit app
page_title = "Multiple Disease Predictor"
page_icon = "🩺"
st.set_page_config(page_title=page_title, page_icon=page_icon, layout="centered")


# Application title and description
st.title(page_title)
st.write(':blue[***Predict your health, one click at a time! 🩺🔮***]')
st.write("The Multiple Disease Predictor app helps you assess your risk for conditions like Diabetes, Heart Disease, "
         "or Parkinson. Simply select a disease, provide some basic health data, and get a quick, accurate prediction "
         "on your condition. Stay ahead of your health with cutting-edge AI! 🩺📊💡")
st.info('Check the About section for source of dataset', icon='ℹ️')
# ---- NAVIGATION MENU -----
selection = option_menu(
    menu_title="Select Disease",
    menu_icon="bi-check2-square",
    options=["Diabetes", "Heart", "Parkinson", "About"],
    icons=["bi-activity", "bi-heart", "bi-person-wheelchair", "app"],  # https://icons.getbootstrap.com
    orientation="horizontal",
)

if selection == "Diabetes":
    # Input Parameters
    st.subheader('Input Parameters')
    with st.expander('User input', expanded=True, icon=':material/settings_input_component:'):
        input_data = diabetes_input_parameters()

    # Model selection
    st.subheader('Select ML Model')
    model_name = st.selectbox('Select the Model', supported_models, label_visibility="collapsed")

if selection == "Heart":
    # Input Parameters
    st.subheader('Input Parameters')
    with st.expander('User input', expanded=True, icon=':material/settings_input_component:'):
        input_data = heart_input_parameters()

    # Model selection
    st.subheader('Select ML Model')
    model_name = st.selectbox('Select the Model', supported_models, label_visibility="collapsed")
if selection == "Parkinson":
    # Input Parameters
    st.subheader('Input Parameters')
    with st.expander('User input', expanded=True, icon=':material/settings_input_component:'):
        input_data = parkinson_input_parameters()

    # Model selection
    st.subheader('Select ML Model')
    model_name = st.selectbox('Select the Model', supported_models, label_visibility="collapsed")

if selection == "About":
    about_app()

