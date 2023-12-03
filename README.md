# Heart Disease Prediction Web Application

## Overview

Heart Disease Prediction is a machine learning project that aims to predict the likelihood of an individual developing heart disease based on various health parameters. This project uses neural networks to build a predictive model. The dataset used for training and testing the model includes features such as age, sex, chest pain type, blood pressure, cholesterol level, and more.

## Project Structure

- `Group8_Heart_Prediction.ipynb`: Jupyter Notebook containing the Python code for the project.
- `heart.csv`: Dataset used for training and testing the model.
- `heart_prediction.h5`: Saved model file.
- `app.py`: Streamlit web application for deploying the model.

## Dependencies
- Python
- TensorFlow
- Scikit-learn
- Pandas
- NumPy
- Seaborn
- Matplotlib

## How to Use

### Running the Jupyter Notebook

1. Open `Group8_Heart_Prediction.ipynb` using Jupyter Notebook or Google Colab.
2. Run each cell sequentially to train the model, perform data analysis, and evaluate the model.

### Deploying the Streamlit Web Application

1. Install the required libraries: `pip install streamlit, scikit-learn, pandas, numpy, keras`.
2. Run the Streamlit application in the terminal: `streamlit run heart_prediction_app.py`.
3. Open a web browser and go to the local host link provided.
4. Enter the required health parameters, and the application will predict the likelihood of heart disease.

## Model Evaluation

The model is evaluated using accuracy, precision, recall, and AUC score. For additional details, refer to the model evaluation section in the Jupyter Notebook.

## Additional Resources: Link to YouTube Video

- [https://youtu.be/R_uE5n_Zlx4] - Watch a video explaining how the Heart Disease Prediction application works.


## Acknowledgments
- This project was developed as part of Ashesi University's coursework in Introduction to AI, Final Project.


## Authors
- Faisal Alidu, Emmanuel Soumahoro

## ---


## Disclaimer

This application is for educational and informational purposes only. It should not be considered as medical advice. Consult with a healthcare professional for accurate medical diagnosis and guidance.

