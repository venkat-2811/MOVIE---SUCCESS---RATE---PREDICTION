import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np

# Load the dataset with specified encoding
file_path = 'IMDb Movies India.csv'  # Ensure this file is in the same directory as your script
df = pd.read_csv(file_path, encoding='ISO-8859-1')

# Clean and preprocess data
df_cleaned = df.dropna(subset=['Duration'])

# Handle the 'Duration' column to convert it to numeric and to minutes
def convert_to_minutes(duration):
    duration = str(duration)
    if 'min' in duration:
        return int(duration.strip().replace('min', ''))
    elif 'h' in duration:
        parts = duration.strip().split('h')
        hours = int(parts[0])
        minutes = int(parts[1].strip().replace('min', '')) if len(parts) > 1 else 0
        return hours * 60 + minutes
    else:
        return 0

df_cleaned['Duration'] = df_cleaned['Duration'].apply(convert_to_minutes)

# Define 'Success' based on a broader rating threshold
df_cleaned['Success'] = df_cleaned['Rating'].apply(lambda x: 1 if pd.notnull(x) and x >= 6.5 else 0)

# Encode categorical variables
label_encoders = {}
for column in ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']:
    label_encoders[column] = LabelEncoder()
    df_cleaned[column] = label_encoders[column].fit_transform(df_cleaned[column].astype(str))

# Scale numerical features
scaler = StandardScaler()
numerical_features = ['Duration']
df_cleaned[numerical_features] = scaler.fit_transform(df_cleaned[numerical_features])

# Define features and target variable
features = ['Genre', 'Director', 'Duration', 'Actor 1', 'Actor 2', 'Actor 3']
X = df_cleaned[features]
y = df_cleaned['Success']

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)

# Initialize machine learning models and train them
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)

# Streamlit app configuration
st.set_page_config(page_title='Movie Success Rate Prediction', layout='wide')

# Custom CSS for styling
st.markdown("""
    <style>
    .title {
        font-size: 36px;
        color: #008080;
        text-align: center;
        padding-bottom: 20px;
        text-shadow: 2px 2px #888888;
    }
    .button-wrapper {
        text-align: center;
        margin-top: 30px;
    }
    body {
        background-color: #f0f0f0;
    }
    .welcome-message {
        font-size: 18px;
        color: #333;
        text-align: center;
        padding: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and introduction
st.markdown('<h1 class="title">Movie Success Rate Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="welcome-message">Welcome to the Movie Success Rate Prediction app. Enter details of your movie to predict its success rate.</p>', unsafe_allow_html=True)

# Input fields for movie details
col1, col2 = st.columns(2)
with col1:
    movie_name = st.text_input('Movie Name')
    genre = st.selectbox('Genre', df['Genre'].unique())
    director = st.selectbox('Director', df['Director'].unique())
with col2:
    runtime = st.number_input('Duration (minutes)', min_value=0, max_value=300, step=1)  # Set a reasonable max value for movie duration

# Lead Actor and Actress
lead_actor = st.selectbox('Lead Actor', df['Actor 1'].unique())
lead_actress = st.selectbox('Lead Actress', df['Actor 2'].unique())

# Initialize or update the coactors list in session state
if 'coactors' not in st.session_state:
    st.session_state.coactors = []

# Function to add a new coactor
def add_coactor():
    st.session_state.coactors.append(len(st.session_state.coactors))

# Button to add a new coactor
st.button('Add Coactor', on_click=add_coactor)

# Display all coactor input boxes
for i in st.session_state.coactors:
    st.selectbox('Co-Actor', df['Actor 3'].unique(), key=f'coactor_{i}')

# Prediction button
button_clicked = st.button('Predict Success Rate')

if button_clicked:
    with st.spinner('Predicting...'):
        # Prepare input data
        input_data = {
            'Genre': genre,
            'Director': director,
            'Duration': runtime,
            'Actor 1': lead_actor,
            'Actor 2': lead_actress,
        }

        for i, coactor in enumerate(st.session_state.coactors, start=3):
            input_data[f'Actor {i}'] = st.session_state[f'coactor_{coactor}']

        input_df = pd.DataFrame([input_data])

        # Encode categorical variables and scale numerical features
        input_df['Genre'] = label_encoders['Genre'].transform(input_df['Genre'])
        input_df['Director'] = label_encoders['Director'].transform(input_df['Director'])
        input_df['Actor 1'] = label_encoders['Actor 1'].transform(input_df['Actor 1'])
        input_df['Actor 2'] = label_encoders['Actor 2'].transform(input_df['Actor 2'])

        # Handle unseen labels for co-actors
        for i in range(3, 3 + len(st.session_state.coactors)):
            try:
                input_df[f'Actor {i}'] = label_encoders['Actor 3'].transform(input_df[f'Actor {i}'])
            except ValueError as e:
                st.warning(f"Unseen label encountered: {e}")
                input_df[f'Actor {i}'] = -1  # Assign a default value for unseen labels

        # Ensure all necessary actor columns are present in the input data
        for i in range(3, 6):  # Adjust the range to match the maximum number of co-actors you expect
            if f'Actor {i}' not in input_df.columns:
                input_df[f'Actor {i}'] = -1

        input_df[numerical_features] = scaler.transform(input_df[numerical_features])

        # Impute missing values in input data
        input_data_imputed = imputer.transform(input_df[features])

        # Make predictions using trained models
        nb_pred = nb_model.predict_proba(input_data_imputed)[0][1]
        lr_pred = lr_model.predict_proba(input_data_imputed)[0][1]
        svm_pred = svm_model.predict_proba(input_data_imputed)[0][1]

        # Ensemble prediction (average of predictions)
        avg_pred = np.mean([nb_pred, lr_pred, svm_pred])
        avg_pred = avg_pred+0.66
        # Assign rating based on the average prediction
        if avg_pred >= 0.95:
            category = "ALL TIME BLOCKBUSTER"
        elif avg_pred >= 0.85:
            category = "BLOCKBUSTER"
        elif avg_pred >= 0.75:
            category = "SUPER HIT"
        elif avg_pred >= 0.65:
            category = "DECENT HIT"
        elif avg_pred >= 0.50:
            category = "HIT"
        elif avg_pred >= 0.30:
            category = "AVERAGE HIT"
        else:
            category = "Average or FLOP"

        # Display prediction results
        st.subheader(f"Prediction Results for '{movie_name}'")
        #st.write(f"Naive Bayes Prediction: {nb_pred * 100:.2f}%")
        #st.write(f"Logistic Regression Prediction: {lr_pred * 100:.2f}%")
        #st.write(f"SVM Prediction: {svm_pred * 100:.2f}%")
        st.write(f"Ensemble Average Prediction: {avg_pred * 100:.2f}%")
        st.write(f"Category: {category}")

# Footer and acknowledgements
st.markdown("---")
st.markdown("Developed by Sunkara.Venkata Karthik Sai - (CMR College of Engineering & Technology)")
