## Import libraries
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from tensorflow.keras.models import load_model


## Load the trained model
model = load_model('heart_prediction.h5')


# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Natembea Clinic (Ashesi) Disease Prediction System',
                          
                          ['Heart Disease Prediction'],
                          icons=['activity'],
                          menu_icon="hospital",
                          default_index=0,

                         styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "green"},
                       }
    )


# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):
    
    # page title
    st.title(':heart: Heart Disease Prediction App')
    #st.title('Heart Disease Prediction App')
    st.markdown("""
    **:dart: Enter patient|user health information to predict heart health status.**
    """)

    
    col1, col2, col3 = st.columns(3)
   
    st.markdown(
        """
        <style>
            .main {
                padding: 8px;
                color: blue;
            }
            .st-bw {
                background-color:  green;
                color: orange;
                padding: 10px;
                border-radius: 10px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    with col1:
        st.markdown('<style>div[data-baseweb="input"]{background-color: green; padding: 5px; border-radius: 10px;}</style>', unsafe_allow_html=True)
        age = st.number_input('Age:', min_value=0, max_value=150, value=0)
        
    with col2:
        sex_mapping = {' ': -1, 'female': 0, 'male': 1}

        # Check if user has made a selection
        selected_sex = st.selectbox('Sex:', list(sex_mapping.keys()))

        # Update sex only if a selection is made
        sex = sex_mapping[selected_sex] if selected_sex != '' else None

        
    with col3:
        cp_mapping = {' ': -1 , 'typical angina': 0, 'atypical angina': 1, 'non-anginal pain': 2, 'asymptomatic': 3}
    
        # Check if user has made a selection
        selected_cp = st.selectbox('Chest Pain types:', list(cp_mapping.keys()))
    
        # Update cp only if a selection is made
        cp = cp_mapping[selected_cp] if selected_cp != '' else None
        
    with col1:
        trtbps = st.number_input('Resting Blood Pressure:', min_value=0, max_value=300, value=0)
        
    with col2:
        chol = st.number_input('Serum Cholestoral in mg/dl:', min_value=0, max_value=800, value=0)
        
    with col3:
        fbs_mapping = {' ': -1 , 'false': 0, 'true': 1}

        # Check if user has made a selection
        selected_fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl:', list(fbs_mapping.keys()))

        # Update fbs only if a selection is made
        fbs = fbs_mapping[selected_fbs] if selected_fbs != '' else None
        
    with col1:
        restecg_mapping = {' ': -1 , 'normal': 0, 'ST-T wave abnormality': 1, 'ventricular hypertrophy': 2}
    
        # Check if user has made a selection
        selected_restecg = st.selectbox('Resting Electrocardiographic results:',list(restecg_mapping.keys()))
    
        # Update restecg only if a selection is made
        restecg = restecg_mapping[selected_restecg] if selected_restecg != '' else None
        
    with col2:
        thalachh = st.number_input('Maximum Heart Rate achieved:', min_value=0, max_value=300, value=0)
        
    with col3:
        exng_mapping = {' ': -1 , 'no': 0, 'yes': 1}

        # Check if user has made a selection
        selected_exng = st.selectbox('Exercise Induced Angina:', list(exng_mapping.keys()))

        # Update exang only if a selection is made
        exng = exng_mapping[selected_exng] if selected_exng != '' else None
        
    with col1:
        oldpeak = st.number_input('ST depression induced by exercise:', min_value=0.00, max_value=10.00, value=0.00, step=0.10)
        
    with col2:
        slp_mapping = {' ': -1 , 'downsloping': 0, 'flat': 1, 'upsloping': 2}
    
        # Check if user has made a selection
        selected_slp = st.selectbox('peak exercise ST segment:', list(slp_mapping.keys()))
    
        # Update slp only if a selection is made
        slp = slp_mapping[selected_slp] if selected_slp != '' else None
        
    with col3:
        caa_mapping = {' ': -1 , 'none': 0, 'mild': 1, 'moderate': 2, 'severe': 3, 'akinesis':4}
    
        # Check if user has made a selection
        selected_caa = st.selectbox('Major vessels colored by flourosopy:', list(caa_mapping.keys()))
    
        # Update caa only if a selection is made
        caa = caa_mapping[selected_caa] if selected_caa != '' else None


        
    with col2:
        thall_mapping = {' ': -1 , 'none': 0, 'normal': 1, 'fixed defect': 2, 'reversable defect': 3}
    
        # Check if user has made a selection
        selected_thall = st.selectbox('thalassemia (a blood disorder):', list(thall_mapping.keys()))
    
        # Update thall only if a selection is made
        thall = thall_mapping[selected_thall] if selected_thall != '' else None
        
        
     
    

   # code for Prediction
heart_diagnosis = ''

# creating a button for Prediction
if st.button('Heart Disease Test Result'):
    # Convert input data to float32
    input_data = np.array([[float(age), float(sex), float(cp), float(trtbps), float(chol), float(fbs),
                             float(restecg), float(thalachh), float(exng), float(oldpeak), float(slp),
                             float(caa), float(thall)]], dtype=np.float32)
    
    prediction = model.predict(input_data)
    output = np.round(prediction[0], 2)
    # st.warning(f"Positive Heart Disease Probability: {output}")

    if output == 1:
        heart_diagnosis = 'Poistive; You have a heart disease'
    else:
        heart_diagnosis = 'Negative; Your heart is healthy'

    st.success(heart_diagnosis)

    # Calculate confidence factor
    confidence_factor = 2.58 * np.sqrt((output * (1 - output)) / 1)  
    st.write(f"Confidence Factor: {confidence_factor}")


        
    