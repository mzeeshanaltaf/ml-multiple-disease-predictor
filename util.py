import streamlit as st


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
        col1, col2, col3 = st.columns(3)
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


# Get heart data from the user
def heart_input_parameters():
    # Dictionaries to convert labels to their corresponding integer values
    gender_dic = {'Male': 1, 'Female': 0}
    chest_paint_dict = {'Asymptomatic': 1, 'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 0}
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
        spread1 = col2.number_input('Spread1', min_value=-7.0, max_value=-2.0, value=-5.5)
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
        st.markdown(''' Source code: [GitHub](https://github.com/mzeeshanaltaf/)''')

