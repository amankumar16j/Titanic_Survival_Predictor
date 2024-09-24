# Import necessary libraries
import streamlit as st
import pandas as pd
import joblib



# st.markdown(page_bg_img, unsafe_allow_html=True)
# st.markdown(page_bg_img, unsafe_allow_html=True)



# Load the Titanic dataset (you might have your own data loading process)
@st.cache
def load_data():
    df = pd.read_csv('train.csv')  # Replace with your dataset path
    return df

# Load the pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = joblib.load('titanic_model.pkl')
    return model

# Main function to run the Streamlit app
def main():
    st.title('Titanic Survival Prediction')


    # Load data
    df = load_data()

    # Load model
    model = load_model()

    # Display dataset
    st.subheader('Titanic Dataset')
    st.write(df)

    # Add user input for prediction
    st.sidebar.title('Make a Prediction')
    # Example inputs (you can customize this based on your model's input requirements)
    passenger_class = st.sidebar.selectbox('Passenger Class', ['1st', '2nd', '3rd'])
    sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])
    age = st.sidebar.number_input('Age', min_value=0, max_value=100, value=30)
    sibsp=st.sidebar.number_input('SibSp',min_value=0,max_value=8)
    parch=st.sidebar.number_input('Parch',min_value=0,max_value=6)
    fare=st.sidebar.number_input('Fare',value=10)
    embarked = st.sidebar.selectbox('Embarked', ['C', 'Q', 'S'])
    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'Pclass': [passenger_class],
        'Sex': [sex],
        'Age': [age],
        'SibSp':[sibsp],
        'Parch':[parch],
        'Fare':[fare],
        'Embarked': [embarked],
    })

    # Convert categorical variables to numerical (you need to apply the same preprocessing as in your model)
    input_data['Pclass'] = input_data['Pclass'].map({'1st': 1, '2nd': 2, '3rd': 3})
    input_data['Sex'] = input_data['Sex'].map({'Male': 1, 'Female': 0})
    input_data['Embarked'] = input_data['Embarked'].map({'C': 2, 'Q': 3, 'S': 1})

    # Predict survival
    if st.sidebar.button('Predict'):
        prediction = model.predict(input_data)
        if prediction[0] == 0:
            st.sidebar.error('Unfortunately, the passenger did not survive.')
        else:
            st.sidebar.success('The passenger survived!')

if __name__ == '__main__':
    main()
