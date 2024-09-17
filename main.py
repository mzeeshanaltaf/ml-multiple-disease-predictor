import streamlit as st
from streamlit_option_menu import option_menu
from util import *
from models import *

# Define session state variables
if "model_name" not in st.session_state:
    st.session_state.model_name = None

# List of supported Machine Learning Models
supported_models = ["Logistic Regression", "Support Vector Machines", 'K-Nearest Neighbor', "Decision Tree",
                    "Random Forest", 'AdaBoost', 'Gradient Boost', 'XGBoost', "Gaussian NB"]

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
st.info('Dataset for the diseases taken from Kaggle. Check the About section in the sidebar for links.', icon='ℹ️')

# Configure sidebar
with st.sidebar:
    # Display Disclaimer
    display_disclaimer()
    # Model selection option
    st.subheader('Select Machine Learning Model')
    st.session_state.model_name = st.selectbox('Select the Model', supported_models, label_visibility="collapsed")

    # About section
    about_app()

    # Footer
    display_footer()

# ---- NAVIGATION MENU -----
selection = option_menu(
    menu_title="Select Disease",
    menu_icon="bi-check2-square",
    options=["Diabetes", "Heart", "Parkinson", 'Liver', 'Kidney'],
    icons=["bi-activity", "bi-heart", "bi-person-wheelchair", "bi-file-medical", "bi-prescription"],
    orientation="horizontal",
    default_index=1,
)

# If selection is Diabetes
if selection == "Diabetes":
    # Input Parameters
    st.subheader('Input Parameters')
    with st.expander('User Input', expanded=True, icon=':material/settings_input_component:'):
        input_data = diabetes_input_parameters()

    # Train the model and get prediction and probability of outcome
    model, scalar, df_performance_metric, cm = train_model(st.session_state.model_name, selection)
    prediction, probability = model_predictions(input_data, model, scalar)
    display_prediction(prediction, probability, selection)

    # Display performance metrics
    display_performance_metrics(df_performance_metric)

    # Display Confusion Matrix
    display_confusion_matrix(cm)

# If selection is Heart
if selection == "Heart":
    # Input Parameters
    st.subheader('Input Parameters')
    with st.expander('User Input', expanded=True, icon=':material/settings_input_component:'):
        input_data = heart_input_parameters()

    # Train the model and get prediction and probability of outcome
    model, scalar, df_performance_metric, cm = train_model(st.session_state.model_name, selection)
    prediction, probability = model_predictions(input_data, model, scalar)
    display_prediction(prediction, probability, selection)

    # Display performance metrics
    display_performance_metrics(df_performance_metric)

    # Display Confusion Matrix
    display_confusion_matrix(cm)

# If selection is Parkinson
if selection == "Parkinson":
    # Input Parameters
    st.subheader('Input Parameters')
    with st.expander('User Input', expanded=True, icon=':material/settings_input_component:'):
        input_data = parkinson_input_parameters()

    # Train the model and get prediction and probability of outcome
    model, scalar, df_performance_metric, cm = train_model(st.session_state.model_name, selection)
    prediction, probability = model_predictions(input_data, model, scalar)
    display_prediction(prediction, probability, selection)

    # Display performance metrics
    display_performance_metrics(df_performance_metric)

    # Display Confusion Matrix
    display_confusion_matrix(cm)

# If selection is Parkinson
if selection == "Liver":
    # Input Parameters
    st.subheader('Input Parameters')
    with st.expander('User Input', expanded=True, icon=':material/settings_input_component:'):
        input_data = liver_input_parameters()

    # Train the model and get prediction and probability of outcome
    model, scalar, df_performance_metric, cm = train_model(st.session_state.model_name, selection)
    prediction, probability = model_predictions(input_data, model, scalar)
    display_prediction(prediction, probability, selection)

    # Display performance metrics
    display_performance_metrics(df_performance_metric)

    # Display Confusion Matrix
    display_confusion_matrix(cm)

# If selection is Parkinson
if selection == "Kidney":
    # Input Parameters
    st.subheader('Input Parameters')
    with st.expander('User Input', expanded=True, icon=':material/settings_input_component:'):
        input_data = kidney_input_parameters()

    # Train the model and get prediction and probability of outcome
    model, scalar, df_performance_metric, cm = train_model(st.session_state.model_name, selection)
    prediction, probability = model_predictions(input_data, model, scalar)
    display_prediction(prediction, probability, selection)

    # Display performance metrics
    display_performance_metrics(df_performance_metric)

    # Display Confusion Matrix
    display_confusion_matrix(cm)




