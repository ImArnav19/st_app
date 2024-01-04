import base64
from sklearn.linear_model import LinearRegression
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import accuracy_score, classification_report
from io import StringIO
import graphviz
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split

from dtreeviz import dtreeviz


st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")



# Load the dataset
@st.cache_resource 
def load_data(uploaded_data):
    data = pd.read_csv(uploaded_data)
    return data

# Function to display basic team information
def display_team_info():
    st.subheader('Team Members')
    st.write('Arnav More - 60003210201')
    st.write('Fahad Siddiqui - 60003210207')
    st.write('Murtaza Shikari - 60003210211')
    
    # Add more team members as needed

def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="DWMProj.csv">Download CSV File</a>'
    return href

def eda_analysis(data):
    

    
    # Display basic statistics
    st.write('**Dataset Overview:**')
    st.write(data.head())

    st.write('**Summary Statistics:**')
    st.write(data.describe())

    col1, col2,col3 = st.columns(3)

    with col1:

        st.write('**Attribute Types:**')
        attribute_types = data.dtypes
        st.write(attribute_types)

    with col2:
        st.write('**Missing Values:**')
        missing_values = data.isnull().sum()
        st.write(missing_values)

    with col3:
        st.write('**Number of Unique Values:**')
        unique_values_count = data.nunique()
        st.write(unique_values_count)

    
    st.header('**Filter Your Data:**')
    unique_values_dict = {}

    for column in data.columns:
        unique_values_dict[column] = data[column].unique()

    for i,j in unique_values_dict.items():
        st.sidebar.header(i)
        var = st.sidebar.multiselect(i,j,j)
        data = data[(data[i].isin(var))]

    st.dataframe(data)
    st.write('Data Dimension: ' + str(data.shape[0]) + ' rows and ' + str(data.shape[1]) + ' columns.')
    st.markdown(filedownload(data), unsafe_allow_html=True)

    # Heatmap
    if st.button('Intercorrelation Heatmap'):
        
        object_columns = data.select_dtypes(include='object').columns
        h_data = data.drop(columns=object_columns)
        st.header('Intercorrelation Matrix Heatmap')
        h_data.to_csv('output.csv',index=False)
        df = pd.read_csv('output.csv')

        corr = df.corr()
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True
        with sns.axes_style("white"):
            f, ax = plt.subplots(figsize=(7, 5))
            ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
        st.pyplot()

# Function to drop specified columns
def drop_columns(data, columns_to_drop):
    cleaned_data = data.drop(columns=columns_to_drop, errors='ignore')
    return cleaned_data

# Function to fill missing values for selected columns
def fill_missing_values(data, fill_method, columns_to_fill):
    if fill_method == 'mean':
        for column in columns_to_fill:
            data[column] = data[column].fillna(data[column].mean())
    elif fill_method == 'median':
        for column in columns_to_fill:
            data[column] = data[column].fillna(data[column].median())
    elif fill_method == 'mode':
        for column in columns_to_fill:
            data[column] = data[column].fillna(data[column].mode().iloc[0])
    else:
        for column in columns_to_fill:
            custom_fill_value = st.text_input(f'Enter custom value for {column}:', '')
            data[column] = data[column].fillna(custom_fill_value)
    return data

# Function to drop duplicate rows
def drop_duplicates(data, columns_to_check):
    cleaned_data = data.drop_duplicates(subset=columns_to_check, keep='first')
    return cleaned_data

def clean(data):

        cleaned_data_main=data

        st.subheader('Drop Columns')
        columns_to_drop = st.multiselect('Select columns to drop:', data.columns)
        if columns_to_drop:
            cleaned_data = drop_columns(data, columns_to_drop)
            st.write('Data after dropping columns:')
            st.write(cleaned_data)
        else:
            st.info('No columns selected to drop.')

        # Fill missing values for selected columns
        st.subheader('Fill Missing Values')
        fill_method = st.selectbox('Select method to fill missing values:', ['mean', 'median', 'mode', 'custom'])
        if fill_method != 'custom':
            columns_to_fill = st.multiselect('Select columns to fill missing values:', data.columns)
            if columns_to_fill:
                cleaned_data = fill_missing_values(cleaned_data, fill_method, columns_to_fill)
                st.write('Data after filling missing values:')
                st.write(cleaned_data)
            else:
                st.info('No columns selected to fill missing values.')
        else:
            st.warning('Select a method other than custom to fill missing values.')

        # Drop duplicate rows
        st.subheader('Drop Duplicates')
        columns_to_check_duplicates = st.multiselect('Select columns to check duplicates:', data.columns)
        if columns_to_check_duplicates:
            cleaned_data = drop_duplicates(cleaned_data, columns_to_check_duplicates)
            st.write('Data after dropping duplicates:')
            st.write(cleaned_data)
            cleaned_data_main = cleaned_data
        else:
            st.info('No columns selected to check duplicates.')

        st.title("Results After Cleaning")
        
        st.write('Data Dimension: ' + str(data.shape[0]) + ' rows and ' + str(data.shape[1]) + ' columns.')
        st.dataframe(data)
        st.markdown(filedownload(cleaned_data_main), unsafe_allow_html=True)

# Function to train a Decision Tree classifier
def train_decision_tree(X_train, y_train, max_depth):
    classifier = DecisionTreeClassifier(max_depth=max_depth)
    classifier.fit(X_train, y_train)
    return classifier



def classify(data):

        
        # Identify features and target variable
        
        # Identify features and target variable
        features = [col for col in data.columns if col != data.columns[-1]]  # All columns except the last one
        target_variable = st.selectbox("Select the Target Variable (y):", data.columns)

        # Perform label encoding on the target variable
        label_encoder = LabelEncoder()
        data[target_variable] = label_encoder.fit_transform(data[target_variable])

        # One-hot encode categorical features
        data_encoded = pd.get_dummies(data, columns=features)

        # Split data into features (X) and target variable (y)
        X = data_encoded.drop(columns=[target_variable])
        y = data_encoded[target_variable]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Decision Tree classifier
        max_depth = st.slider("Select max depth for Decision Tree:", min_value=1, max_value=10, value=3)
        classifier = train_decision_tree(X_train, y_train, max_depth)

        # Display the decision tree using Plotly
        st.subheader("Decision Tree Visualization:")
        st.write("Note: This visualization is created using Plotly.")

        # Export decision tree rules
        tree_rules = export_text(classifier, feature_names=X.columns.tolist())
        st.text("Decision Tree Rules:\n" + tree_rules)

        # Visualize the decision tree using Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=classifier.feature_importances_,
            y=X.columns,
            mode='markers',
            marker=dict(color='blue', size=12),
            text=X.columns,
            showlegend=False
        ))
        fig.update_layout(title="Decision Tree Feature Importance",
                          xaxis_title="Importance",
                          yaxis_title="Feature",
                          template="plotly_white")
        st.plotly_chart(fig)

       
# Function to train a Linear Regression model
def train_linear_regression(X, y):
    X = X.reshape(-1, 1)  # Reshape X to be 2D

    
    model = LinearRegression()
    model.fit(X, y)
    return model    
        

def regression(data):
    # Sidebar options
        st.sidebar.subheader("Regression Options")
        regression_type = st.sidebar.selectbox("Select Regression Type:", ["Linear Regression"])
        dependent_variable = st.sidebar.selectbox("Select Dependent Variable (Y):", data.columns)

        # Select independent variables
        independent_variables = st.multiselect("Select Independent Variables (X):", [col for col in data.columns if col != dependent_variable])

        # Prepare data for regression
        X = data[independent_variables].values.reshape(-1, len(independent_variables))
        y = data[dependent_variable].values

        # Train Regression model
        model = train_linear_regression(X, y)

        # Display regression equation
        st.subheader("Regression Equation:")
        equation = f"{dependent_variable} = {model.intercept_:.2f}"
        for i, col in enumerate(independent_variables):
            equation += f" + {model.coef_[i]:.2f} * {col}"

        st.write(equation)

        # Plot regression graph
        st.subheader("Regression Plot:")
        plt.figure(figsize=(8, 6))

        # Scatter plot
        sns.scatterplot(x=data[dependent_variable], y=model.predict(X), label="Predicted")
        sns.scatterplot(x=data[dependent_variable], y=data[dependent_variable], label="Actual")

        plt.title("Regression Plot")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.legend()
        st.pyplot()

        # Show regression analysis
        show_analysis = st.checkbox("Show Regression Analysis")
        if show_analysis:
            st.subheader("Regression Analysis:")
            residuals = y - model.predict(X)

            # Residuals vs Fitted Values Plot
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=model.predict(X), y=residuals)
            plt.title("Residuals vs Fitted Values")
            plt.xlabel("Fitted Values")
            plt.ylabel("Residuals")
            st.pyplot()

            # Distribution of Residuals Plot
            plt.figure(figsize=(8, 6))
            sns.histplot(residuals, kde=True)
            plt.title("Distribution of Residuals")
            plt.xlabel("Residuals")
            plt.ylabel("Frequency")
            st.pyplot()



    


# Home Page
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

st.sidebar.header('**DWM** Mini Project')
page = st.sidebar.selectbox('Go to', ['Home', 'EDA', 'Cleaning', 'Classification', 'Regression'])

if page == 'Home':
    # st.image(image, width = 110)
    st.header(' **DWM Mini Project** ')
    message = st.chat_message("assistant")
    message.write("Welcome, Prof. Harshal Dalvi")
    message.bar_chart(np.random.randn(30, 3))

    

    
    display_team_info()
    st.subheader('Upload Dataset')
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    
    
        
        

         

    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.success('File uploaded successfully!')
else:
    # Check if dataset is uploaded
    if st.session_state.uploaded_file is None:
        st.warning('Please upload a dataset on the Home page.')

    # Load the dataset if uploaded
    else:
        data = load_data(st.session_state.uploaded_file)

        

        st.title(page)
        st.header('Dataset Imported')
        st.dataframe(data)

        
        

        # Data Analysis for EDA, Cleaning, Classification, and Regression
        if page == 'EDA':
            st.subheader('Exploratory Data Analysis (EDA)')


            eda_analysis(data)

        elif page == 'Cleaning':
            st.subheader('Data Cleaning')
            # Add your data cleaning code here

            clean(data)

        elif page == 'Classification':
            st.subheader('Classification Analysis')
            # Add your classification analysis code here
            classify(data)
        elif page == 'Regression':
            st.subheader('Regression Analysis')

            regression(data)
            # Add your regression analysis code here

# Display Streamlit app




