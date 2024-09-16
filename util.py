import streamlit as st
import pandas as pd


# Get diabetes data from the user
def diabetes_input_parameters():
    # Get diabetes related input parameters from user
    input_dict = {}
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        age = col1.slider('Age (Years)', min_value=20, max_value=90, value=40)
        pregnancies = col2.slider('Pregnancies', min_value=0, max_value=20, value=2)
        glucose = col3.slider('Glucose (mg/dL)', min_value=40, max_value=200, value=80)

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        bp = col1.slider('Blood Pressure (mm Hg)', min_value=20, max_value=122, value=80)
        skin_thickness = col2.slider('Skin Thickness (mm)', min_value=5, max_value=100, value=30)
        insulin = col3.slider('Insulin (mu U/ml)', min_value=14, max_value=1000, value=150)

    with st.container(border=True):
        col1, col2 = st.columns(2)
        bmi = col1.slider('BMI', min_value=15, max_value=70, value=30)
        dpf = col2.slider('Diabetes Pedigree Function', min_value=0.05, max_value=2.50, value=0.5)

    # Update the input dictionary with user selected values
    input_dict['pregnancies'] = pregnancies
    input_dict['glucose'] = glucose
    input_dict['blood_pressure'] = bp
    input_dict['skin_thickness'] = skin_thickness
    input_dict['insulin'] = insulin
    input_dict['bmi'] = bmi
    input_dict['dpf'] = dpf
    input_dict['age'] = age

    return input_dict


# Get liver data from the user
def liver_input_parameters():
    # Dictionaries to convert labels to their corresponding integer values
    gender_dic = {'Male': 1, 'Female': 0}

    # Get liver related input parameters from user
    input_dict = {}
    with st.container(border=True):
        col1, col2 = st.columns(2)
        age = col1.slider('Age (Years)', min_value=4, max_value=90, value=40)
        gender = col2.selectbox('Gender', ('Male', 'Female'))

    with st.container(border=True):
        col1, col2 = st.columns(2)
        total_bilirubin = col1.number_input('Total Bilirubin (mg/dL)', min_value=0.1, max_value=25.0, value=7.3)
        direct_bilirubin = col2.number_input('Direct Bilirubin (mg/dL)', min_value=0.0, max_value=10.0, value=2.1)

    with st.container(border=True):
        col1, col2 = st.columns(2)
        alkaline_phosphatase = col1.number_input('Alkaline Phosphatase (IU/L)', min_value=40, max_value=900, value=100)
        alanine_aminotransferase = col2.number_input('Alanine Aminotransferase (IU/L)', min_value=5, max_value=1000, value=35)

    with st.container(border=True):
        col1, col2 = st.columns(2)
        aspartate_aminotransferase = col1.number_input('Aspartate Aminotransferase (IU/L)', min_value=10, max_value=1000,
                                                 value=50)
        total_proteins = col2.number_input('Total Proteins (g/dL)', min_value=2.7, max_value=9.6, value=7.0)

    with st.container(border=True):
        col1, col2 = st.columns(2)
        albumin = col1.number_input('Albumin (g/dL)', min_value=0.9, max_value=5.5, value=3.2)
        albumin_globulin_ratio = col2.number_input('Albumin Globulin Ratio', min_value=0.3, max_value=2.8, value=1.0)

    # Update the input dictionary with user selected values
    input_dict['age'] = age
    input_dict['gender'] = gender_dic[gender]
    input_dict['total_bilirubin'] = total_bilirubin
    input_dict['direct_bilirubin'] = direct_bilirubin
    input_dict['alkaline_phosphotase'] = alkaline_phosphatase
    input_dict['alamine_aminotransferase'] = alanine_aminotransferase
    input_dict['aspartate_aminotransferase'] = aspartate_aminotransferase
    input_dict['total_proteins'] = total_proteins
    input_dict['albumin'] = albumin
    input_dict['albumin_globulin_ratio'] = albumin_globulin_ratio

    return input_dict


# Get kidney data from the user
def kidney_input_parameters():
    # Dictionaries to convert labels to their corresponding integer values
    rbc_dict = {'Abnormal': 0, 'Normal': 0}
    pus_cells_dict = {'Abnormal': 0, 'Normal': 0}
    pus_cells_clumps_dict = {'Not Present': 0, 'Present': 1}
    bacteria_dict = {'Not Present': 0, 'Present': 1}
    hypertension_dict = {'No': 0, 'Yes': 1}
    diabetes_mellitus_dict = {'No': 0, 'Yes': 1}
    coronary_artery_disease_dict = {'No': 0, 'Yes': 1}
    appetite_dict = {'Good': 0, 'Poor': 1}
    peda_edema_dict = {'No': 0, 'Yes': 1}
    anemia_dict = {'No': 0, 'Yes': 1}

    # Get kidney related input parameters from user
    input_dict = {}
    with st.container(border=True):
        col1, col2 = st.columns(2)
        age = col1.slider('Age (Years)', min_value=2, max_value=90, value=40)
        blood_pressure = col2.number_input('Blood Pressure', min_value=50.0, max_value=180.0, value=76.0)
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        specific_gravity = col1.number_input('Specific Gravity', min_value=1.005, max_value=1.025, value=1.015)
        albumin = col2.number_input('Albumin', min_value=0.0, max_value=5.0, value=1.0)
        sugar = col3.number_input('Sugar', min_value=0.0, max_value=5.0, value=0.5)
    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
        red_blood_cells = col1.selectbox('Red Blood Cells', ('Normal', 'Abnormal'))
        pus_cells = col2.selectbox('Pus Cells', ('Normal', 'Abnormal'))
        pus_cells_clumps = col3.selectbox('Pus Cells Clumps', ('Not Present', 'Present'))
        bacteria = col4.selectbox('Bacteria', ('Not Present', 'Present'))
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        blood_glucose_random = col1.number_input('Blood Glucose Random', min_value=22.0, max_value=500.0, value=150.0)
        blood_urea = col2.number_input('Blood Urea', min_value=1.5, max_value=400.0, value=57.5)
        serum_creatinine = col3.number_input('Serum Creatinine', min_value=0.4, max_value=76.0, value=3.0)
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        sodium = col1.number_input('Sodium', min_value=4.5, max_value=165.0, value=137.0)
        potassium = col2.number_input('Potassium', min_value=2.5, max_value=47.0, value=4.5)
        haemoglobin = col3.number_input('Haemoglobin', min_value=3.0, max_value=18.0, value=12.5)
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        packed_cell_volume = col1.number_input('Packed Cell Volume', min_value=9.0, max_value=54.0, value=35.0)
        white_blood_cell_count = col2.number_input('White Blood Cell Count', min_value=2200.0, max_value=27000.0, value=9000.0)
        red_blood_cell_count = col3.number_input('Red Blood Cell Count', min_value=2.0, max_value=8.0, value=4.5)
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        hypertension = col1.selectbox('hypertension', ('No', 'Yes'))
        diabetes_mellitus = col2.selectbox('Diabetes Mellitus', ('No', 'Yes'))
        coronary_artery_disease = col3.selectbox('Coronary Artery Disease', ('No', 'Yes'))
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        appetite = col1.selectbox('Appetite', ('Good', 'Poor'))
        peda_edema = col2.selectbox('Peda Edema', ('No', 'Yes'))
        anemia = col3.selectbox('Anemia', ('No', 'Yes'))

    # Update the input dictionary with user selected values
    input_dict['age'] = age
    input_dict['blood_pressure'] = blood_pressure
    input_dict['specific_gravity'] = specific_gravity
    input_dict['albumin'] = albumin
    input_dict['sugar'] = sugar
    input_dict['red_blood_cells'] = rbc_dict[red_blood_cells]
    input_dict['pus_cells'] = pus_cells_dict[pus_cells]
    input_dict['pus_cells_clumps'] = pus_cells_clumps_dict[pus_cells_clumps]
    input_dict['bacteria'] = bacteria_dict[bacteria]
    input_dict['blood_glucose_random'] = blood_glucose_random
    input_dict['blood_urea'] = blood_urea
    input_dict['serum_creatinine'] = serum_creatinine
    input_dict['sodium'] = sodium
    input_dict['potassium'] = potassium
    input_dict['haemoglobin'] = haemoglobin
    input_dict['packed_cell_volume'] = packed_cell_volume
    input_dict['white_blood_cell_count'] = white_blood_cell_count
    input_dict['red_blood_cell_count'] = red_blood_cell_count
    input_dict['hypertension'] = hypertension_dict[hypertension]
    input_dict['diabetes_mellitus'] = diabetes_mellitus_dict[diabetes_mellitus]
    input_dict['coronary_artery_disease'] = coronary_artery_disease_dict[coronary_artery_disease]
    input_dict['appetite'] = appetite_dict[appetite]
    input_dict['peda_edema'] = peda_edema_dict[peda_edema]
    input_dict['anemia'] = anemia_dict[anemia]

    return input_dict


# Get heart data from the user
def heart_input_parameters():
    # Dictionaries to convert labels to their corresponding integer values
    gender_dic = {'Male': 1, 'Female': 0}
    chest_paint_dict = {'Asymptomatic': 0, 'Typical Angina': 1, 'Atypical Angina': 2, 'Non-anginal Pain': 3}
    fasting_sugar_dict = {'True': 1, 'False': 0}
    resting_ecg_dict = {'Normal': 0, 'ST-TWave Abnormality': 1, 'LVH': 2}
    angina_dict = {'Yes': 1, 'No': 0}
    slope_dict = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
    thallium_dict = {'Normal': 1, 'Fixed Defect': 2, 'Reversible Defect': 3}
    no_of_vessels_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4}

    # Get heart related input parameters from user
    input_dict = {}
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        age = col1.slider('Age (Years)', min_value=20, max_value=90, value=40)
        sex = col2.selectbox('Gender', ('Male', 'Female'))
        cp = col3.selectbox('Chest Pain Type',
                            ('Asymptomatic', 'Typical Angina', 'Atypical Angina', 'Non-anginal Pain'))

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        rbp = col1.slider('Resting BP (mm Hg)', min_value=40, max_value=200, value=120)
        cholesterol = col2.slider('Serum Cholesterol (mg/dl)', min_value=100, max_value=600, value=200)
        fbs = col3.selectbox('Fasting Blood Sugar > 120 (mg/dl)', ('True', 'False'))

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        resting_ecg = col1.selectbox('Resting ECG', ('Normal', 'ST-TWave Abnormality', 'LVH'))
        max_hr = col2.number_input('Max Heart Rate', min_value=40, max_value=205, value=150)
        ang = col3.selectbox('Angina', ('Yes', 'No'))

    with st.container(border=True):
        col1, col2 = st.columns(2)
        old_peak = col1.number_input('ST Depression induced by exercise', min_value=0.0, max_value=7.0, value=2.5)
        slope = col2.selectbox('Slope', ('Upsloping', 'Flat', 'Downsloping'))
    with st.container(border=True):
        col1, col2 = st.columns(2)
        ca = col1.selectbox('No. of Major Vessels colored by Fluoroscopy', ('0', '1', '2', '3', '4'))
        thallium = col2.selectbox('Thallium Stress Test', ('Normal', 'Fixed Defect', 'Reversible Defect'))

    # Update the input dictionary with user selected values
    input_dict['Age'] = age
    input_dict['Sex'] = gender_dic[sex]
    input_dict['CP'] = chest_paint_dict[cp]
    input_dict['rbp'] = rbp
    input_dict['chol'] = cholesterol
    input_dict['fbs'] = fasting_sugar_dict[fbs]
    input_dict['recg'] = resting_ecg_dict[resting_ecg]
    input_dict['mhr'] = max_hr
    input_dict['angina'] = angina_dict[ang]
    input_dict['old_peak'] = old_peak
    input_dict['slope'] = slope_dict[slope]
    input_dict['ca'] = no_of_vessels_dict[ca]
    input_dict['thal'] = thallium_dict[thallium]

    return input_dict


# Get parkinson data from the user
def parkinson_input_parameters():
    # Get parkinson related input parameters from user
    input_dict = {}
    with st.container(border=True):
        st.write('Vocal Fundamental Frequency:')
        col1, col2, col3 = st.columns(3)
        freq_avg = col1.number_input('Average', min_value=80, max_value=280, value=150)
        freq_hi = col2.number_input('Maximum', min_value=100, max_value=600, value=197)
        freq_lo = col3.number_input('Minimum', min_value=60, max_value=250, value=115)

    with st.container(border=True):
        st.write('Variation in Fundamental Frequency:')
        col1, col2, col3, col4, col5 = st.columns(5)
        jitter_percent = col1.number_input('MDVP:Jitter(%)', min_value=0.001, max_value=0.008, value=0.003)
        jitter_abs = col2.number_input('MDVP:Jitter(Abs)', min_value=0.000007, max_value=0.0003, value=0.00004)
        mdvp_rap = col3.number_input('MDVP:RAP', min_value=0.0006, max_value=0.03, value=0.003)
        mdvp_ppq = col4.number_input('MDVP:PPQ', min_value=0.0008, max_value=0.02, value=0.003)
        jitter_ddp = col5.number_input('Jitter:DDP', min_value=0.002, max_value=0.07, value=0.009)

    with st.container(border=True):
        st.write('Variation in Amplitude:')
        col1, col2, col3 = st.columns(3)
        shimmer = col1.number_input('MDVP:Shimmer', min_value=0.009, max_value=0.12, value=0.03)
        shimmer_db = col2.number_input('MDVP:Shimmer(dB)', min_value=0.08, max_value=1.5, value=0.25)
        shimmer_apq3 = col3.number_input('Shimmer:APQ3', min_value=0.004, max_value=0.06, value=0.01)

        col1, col2, col3 = st.columns(3)
        shimmer_apq5 = col1.number_input('Shimmer:APQ5', min_value=0.005, max_value=0.08, value=0.015)
        mdvp_apq = col2.number_input('MDVP:APQ', min_value=0.007, max_value=0.15, value=0.025)
        shimmer_dda = col3.number_input('Shimmer:DDA', min_value=0.01, max_value=0.17, value=0.05)

    with st.container(border=True):
        st.write('Ratio of noise to tonal components in the voice:')
        col1, col2 = st.columns(2)
        nhr = col1.number_input('NHR', min_value=0.0006, max_value=0.35, value=0.025)
        hnr = col2.number_input('HNR', min_value=8.0, max_value=35.0, value=22.0)

    with st.container(border=True):
        st.write('Dynamic Complexity Measures:')
        col1, col2 = st.columns(2)
        rpde = col1.number_input('RPDE', min_value=0.2, max_value=0.7, value=0.5)
        d2 = col2.number_input('D2', min_value=1.2, max_value=4.0, value=2.5)

    with st.container(border=True):
        st.write('Signal Fractal Scaling Exponent / Fundamental Freq Variation:')
        col1, col2 = st.columns(2)
        dfa = col1.number_input('DFA', min_value=0.5, max_value=0.9, value=0.72)
        spread1 = col2.number_input('Spread1', min_value=-8.0, max_value=-2.0, value=-5.5)
        spread2 = col1.number_input('Spread2', min_value=0.006, max_value=0.5, value=0.22)
        ppe = col2.number_input('PPE', min_value=0.04, max_value=0.6, value=0.2)

        # Update the input dictionary with user selected values
        input_dict['freq_avg'] = freq_avg
        input_dict['freq_hi'] = freq_hi
        input_dict['freq_lo'] = freq_lo
        input_dict['jitter_percent'] = jitter_percent
        input_dict['jitter_abs'] = jitter_abs
        input_dict['mdvp_rap'] = mdvp_rap
        input_dict['mdvp_ppq'] = mdvp_ppq
        input_dict['jitter_ddp'] = jitter_ddp
        input_dict['shimmer'] = shimmer
        input_dict['shimmer_db'] = shimmer_db
        input_dict['shimmer_apq3'] = shimmer_apq3
        input_dict['shimmer_apq5'] = shimmer_apq5
        input_dict['mdvp_apq'] = mdvp_apq
        input_dict['shimmer_dda'] = shimmer_dda
        input_dict['nhr'] = nhr
        input_dict['hnr'] = hnr
        input_dict['rpde'] = rpde
        input_dict['dfa'] = dfa
        input_dict['spread1'] = spread1
        input_dict['spread2'] = spread2
        input_dict['d2'] = d2
        input_dict['ppe'] = ppe

        return input_dict


# Function to display the model prediction and probability
def display_prediction(prediction, probability, selection):
    st.subheader("Prediction")
    if prediction[0] == 0:
        prob = "{:.3f}".format(probability[0][0])
        st.success(f"{selection} Disease: NO")
    else:
        prob = "{:.3f}".format(probability[0][0])
        st.error(f"{selection} Disease: YES")


# Function to display the footer
def display_performance_metrics(df_performance_metric):
    st.subheader('Performance Metrics')
    accuracy_col, f1_score_col, precision_col, recall_col, roc_auc_score_col = st.columns(5)
    with accuracy_col:
        with st.container(border=True):
            st.metric('*Accuracy*', value=df_performance_metric['Accuracy'].iloc[0])
    with f1_score_col:
        with st.container(border=True):
            st.metric('*F1 Score*', value=float(df_performance_metric['F1 Score'].iloc[0]))
    with precision_col:
        with st.container(border=True):
            st.metric('*Precision*', value=float(df_performance_metric['Precision'].iloc[0]))
    with recall_col:
        with st.container(border=True):
            st.metric('*Recall*', value=float(df_performance_metric['Recall'].iloc[0]))
    with roc_auc_score_col:
        with st.container(border=True):
            st.metric('*ROC AUC Score*', value=float(df_performance_metric['ROC AUC Score'].iloc[0]))


def display_confusion_matrix(cm):
    st.subheader('Prediction Outcome Table')
    predicted_positive = [cm[0][0], cm[1][0]]
    predicted_negative = [cm[0][1], cm[1][1]]

    # Data for confusion matrix
    data = {'Predicted Positive': predicted_positive,  # [True Positive, False Positive]
            'Predicted Negative': predicted_negative}  # [False Negative, True Negative]

    # Creating DataFrame
    df_confusion_matrix = pd.DataFrame(data, index=['Actual Positive', 'Actual Negative'])
    st.dataframe(df_confusion_matrix)


def display_disclaimer():
    st.subheader('Disclaimer')
    st.info('*This application is for information purpose only and should not be considered as medical '
            'advice or a conclusive diagnosis. Always consult a qualified healthcare professional for an accurate '
            'diagnosis and personalized medical advice.*')


def about_app():
    st.subheader('About')
    with st.expander('Application'):
        st.markdown(''' Multiple Disease Predictor''')
    with st.expander('Supported Machine Learning Models'):
        st.markdown(''' 
        * Logistic Regression 
        * Support Vector Machine
        * Decision Tree 
        * Random Forest
        * Gaussian NB
        ''')
    with st.expander('Technologies Used'):
        st.markdown(''' 
        * numpy -- Numerical operations
        * pandas -- Data manipulation and analysis
        * scikit_learn -- For machine learning
        * streamlit -- Front end
        ''')
    with st.expander('Data Source'):
        st.markdown(''' 
        * [Diabetes Disease](https://www.kaggle.com/datasets/mathchi/diabetes-data-set)
        * [Heart Disease](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
        * [Parkinson Disease](https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set)
        ''')
    with st.expander('Contact'):
        st.markdown(''' Any Queries: Contact [Zeeshan Altaf](mailto:zeeshan.altaf@92labs.ai)''')
    with st.expander('Source Code'):
        st.markdown(''' Source code: [GitHub](https://github.com/mzeeshanaltaf/ml-multiple-disease-predictor)''')


def random_value_imputation(df, feature):
    random_sample = df[feature].dropna().sample(df[feature].isna().sum())
    random_sample.index = df[df[feature].isnull()].index
    df.loc[df[feature].isnull(), feature] = random_sample
    return df


def impute_mode(df, feature):
    mode = df[feature].mode()[0]
    df[feature] = df[feature].fillna(mode)
    return df


def impute_mean(df, feature):
    mean = df[feature].mean()
    df[feature] = df[feature].fillna(mean)
    return df


def display_footer():
    footer = """
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: transparent;
            text-align: center;
            color: grey;
            padding: 10px 0;
        }
        </style>
        <div class="footer">
            Made with ❤️ by <a href="mailto:zeeshan.altaf@92labs.ai">Zeeshan</a>.
            Source code <a href='https://github.com/mzeeshanaltaf/ml-multiple-disease-predictor'>here</a>.</div> 
        </div>
    """
    st.markdown(footer, unsafe_allow_html=True)
