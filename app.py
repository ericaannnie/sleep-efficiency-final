
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics as mt
import streamlit as st
import random
from PIL import Image
import altair as alt


# Image
image_sal = Image.open('sal1.jpg')
st.image(image_sal, width=500, use_column_width=True)

# Title
st.title("Employment Salary Prediction")

# Sidebar header and a separator
st.sidebar.header("Dashboard")
st.sidebar.markdown("---")

# Dropdown menu for selecting the page mode (Introduction, Visualization, Prediction, Deployment)
app_mode = st.sidebar.selectbox('🔎 Select Page',['Introduction','Visualization','Prediction'])

# Dropdown menu for selecting the dataset (currently only "Salary" is available)
select_dataset =  st.sidebar.selectbox('💾 Select Dataset',["Salary"])

# Load the salary quality dataset
df_temp = pd.read_csv("Salary_Data.csv")
#####################################################################
# Changes made to data
df = df_temp.dropna()

df = df[df.Gender != 'Other']

df['Education Level'] = df['Education Level'].replace(["Bachelor's Degree", "Bachelor's"], "Bachelor's")
df['Education Level'] = df['Education Level'].replace(["Master's Degree", "Master's"], "Master's")
df['Education Level'] = df['Education Level'].replace(["Phd", "phD"], "PhD")

df.groupby('Job Title').filter(lambda x : len(x)>30)

# Create new df just in case
df2 = df[['Age', 'Years of Experience', 'Salary']].copy()

#####################################################################






# Dropdown menu for selecting which variable from the dataset to predict
list_var = df2.columns
select_variable =  st.sidebar.selectbox('🎯 Select Variable to Predict',['Salary'])

# Introduction page content
if app_mode == 'Introduction':
    # Display dataset details
    st.markdown("### 00 - Show Dataset")

#############
    # Split the page into 6 columns to display information about each salary variable
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    # CSS to style the columns
    box_style = """
        border: 2px solid #FFFFFF;
        padding: 10px; 
        margin: 5px;
        border-radius: 10px; 
        width: 110px;
        height: 220px;
        font-size: 15px;
    """

    # Apply to each column
    col1.markdown(f'<div style="{box_style}"><strong>Age</strong><br><br><br>The age at which the person currently is.</div>', unsafe_allow_html=True)
    col2.markdown(f'<div style="{box_style}"><strong>Gender</strong><br><br><br>Gender of the person. Either male or female.</div>', unsafe_allow_html=True)
    col3.markdown(f'<div style="{box_style}"><strong>Education Level</strong><br><br>Academic education level of the professional.<br>Ex: Bachelors</div>', unsafe_allow_html=True)
    col4.markdown(f'<div style="{box_style}"><strong>Job Title</strong><br><br><br>Title of the professional\'s job.<br>Ex: Project Manager</div>', unsafe_allow_html=True)
    col5.markdown(f'<div style="{box_style}"><strong>Years of Experience</strong><br><br>Number of years in the workforce.</div>', unsafe_allow_html=True)
    col6.markdown(f'<div style="{box_style}"><strong>Salary</strong><br><br><br>Salary in USD of the professional.</div>', unsafe_allow_html=True)
##############

# Split the page into 10 columns to display information about each wine quality variable
    #col1, col2, col3, col4, col5, col6 = st.columns(6)

    # Descriptions for each variable in the dataset
    # ... [The code here provides descriptions for each wine quality variable]


    #col1.markdown(" **Age** ")
    #col1.markdown("Age at which the person is currently")
    #col2.markdown(" **Gender** ")
    #col2.markdown("Gender of the person. Either male or female.")
    #col3.markdown(" **Education Level** ")       
    #col3.markdown("Education level of the professional. Ex: Bachelors")
    #col4.markdown(" **Job Title** ")       
    #col4.markdown("Title of the professional's job. Ex: Project Manager")
    #col5.markdown(" **Years of Experience** ")
    #col5.markdown("Number of years in the workforce")
    #col6.markdown(" **Salary** ")
    #col6.markdown("Salary in USD of the professional. ")
    " "
    " "


    # Allow users to view either the top or bottom rows of the dataset
    num = st.number_input('No. of Rows', 5, 10)
    head = st.radio('View from top (head) or bottom (tail)', ('Head', 'Tail'))
    if head == 'Head':
        st.dataframe(df.head(num))
    else:
        st.dataframe(df.tail(num))

    # Display the shape (number of rows and columns) of the dataset
    st.markdown("Number of rows and columns helps us to determine how large the dataset is.")
    st.text('(Rows,Columns)')
    st.write(df.shape)


    st.markdown("### 01 - Description")
    st.dataframe(df.describe())



    st.markdown("### 02 - Missing Values")
    st.markdown("Missing values are known as null or NaN values. Missing data tends to **introduce bias that leads to misleading results.**")
    dfnull = df_temp.isnull().sum()/len(df_temp)*100
    totalmiss = dfnull.sum().round(2)
    st.write("Percentage of total missing values:",totalmiss)
    st.write(dfnull)
    if totalmiss <= 30:
        st.success("Looks good! as we have less then 30 percent of missing values.")
    else:
        st.warning("Poor data quality due to greater than 30 percent of missing value.")
        st.markdown(" > Theoretically, 25 to 30 percent is the maximum missing values are allowed, there's no hard and fast rule to decide this threshold. It can vary from problem to problem.")

    st.markdown("### 03 - Completeness")
    st.markdown(" Completeness is defined as the ratio of non-missing values to total records in dataset.")
    # st.write("Total data length:", len(df))
    nonmissing = (df_temp.notnull().sum().round(2))
    completeness= round(sum(nonmissing)/len(df_temp),2)
    st.write("Completeness ratio:",completeness)
    st.write(nonmissing)
    if completeness >= 0.80:
        st.success("Looks good! as we have completeness ratio greater than 0.85.")

    else:
        st.success("Poor data quality due to low completeness ratio( less than 0.85).")


if app_mode == 'Visualization':

    list_var = df2.columns
    # Display a header for the Visualization section
    st.markdown("## Visualization")

    # Allow users to select two variables from the dataset for visualization
    symbols = st.multiselect("Select two variables", list_var, [ "Years of Experience", "Salary"])

    # Create a slider in the sidebar for users to adjust the plot width
    width1 = st.sidebar.slider("plot width", 1, 25, 10)

    # Create tabs for different types of visualizations
    tab1, tab2 = st.tabs(["Line Chart", "📈 Correlation"])

    # Content for the "Line Chart" tab
    tab1.subheader("Line Chart")
    # Display a line chart for the selected variables
    st.line_chart(data=df2, x=symbols[0], y=symbols[1], width=0, height=0, use_container_width=True)
    # Display a bar chart for the selected variables
    st.bar_chart(data=df2, x=symbols[0], y=symbols[1], use_container_width=True)

    # Content for the "Correlation" tab
    tab2.subheader("Correlation Tab 📉")
    # Create a heatmap to show correlations between variables in the dataset
    fig, ax = plt.subplots(figsize=(width1, width1))
    sns.heatmap(df2.corr(), cmap=sns.cubehelix_palette(8), annot=True, ax=ax)
    tab2.write(fig)

    # Display a pairplot for the first five variables in the dataset
    st.markdown("### Pairplot")
    df3 = df2

    fig3 = sns.pairplot(df3)
    st.pyplot(fig3)



# Check if the app mode is set to 'Prediction'
if app_mode == 'Prediction':
    # Display a header for the Prediction section
    st.markdown("## Prediction")

    # Allow users to adjust the size of the training dataset using a slider in the sidebar
    test_size = st.sidebar.number_input("Train Set Size", min_value=0.00, step=0.01, max_value=1.00, value=0.70)

    # Drop the selected variable from the dataset to prepare for prediction
    new_df = df.drop(labels=select_variable, axis=1)
    list_var = df2.columns

    # Allow users to select explanatory variables for prediction
    output_multi = st.multiselect("Select Explanatory Variables",  ["Years of Experience", "Age"])

    # Define a function to perform linear regression prediction
    def predict(target_choice, test_size, new_df, output_multi):
        """
        This function performs linear regression prediction.
    
        Parameters:
        - target_choice: The target variable to be predicted.
        - train_size: The proportion of the dataset to include in the training set.
        - new_df: The dataframe without the target variable.
        - output_multi: The explanatory variables selected by the user.
    
        Returns:
        - X_train, X_test: Training and testing data.
        - y_train, y_test: Training and testing target values.
        - predictions: Predicted values for the test set.
        - x, y: Full dataset split into explanatory variables and target variable.
        """

        # Select the explanatory variables based on user input
       ####################################################################################
        X = df[['Age','Gender','Education Level', 'Job Title','Years of Experience']]
        
        
        y = df['Salary']
    
        ####################################################################################
        #new_df2 = new_df[output_multi]
       # x = new_df2
        #y = df[target_choice]
    
        # Display the top 25 rows of the explanatory and target variables in the Streamlit app
        col1, col2 = st.columns(2)
        col1.subheader("Feature Columns top 25")
        col1.write(X.head(25))
        col2.subheader("Target Column top 25")
        col2.write(y.head(25))
    
        # Split the data into training and testing sets
        X = pd.get_dummies(data=X, drop_first=True)
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=40)
    
        # Initialize and train a linear regression model
        lm = LinearRegression()
        lm.fit(X_train,y_train)
    
        # Predict the target variable for the test set
        predictions = lm.predict(X_test)
    
        return X_train, X_test, y_train, y_test, predictions, X, y
    
    # Call the prediction function and store the results
    X_train, X_test, y_train, y_test, predictions, X, y = predict(select_variable, test_size, df, list_var)
    
    # Display the results header in the Streamlit app
    st.subheader('🎯 Results')
    
    # Display prediction metrics
    st.write("1) The model explains,", np.round(mt.explained_variance_score(y_test, predictions)*100,2),"% variance of the target feature")
    st.write("2) The Mean Absolute Error of model is:", np.round(mt.mean_absolute_error(predictions,y_test  ),2))
    st.write("3) MSE: ", np.round(mt.mean_squared_error(predictions,y_test ),2))
    st.write("4) The R-Square score of the model is " , np.round(mt.r2_score(predictions, y_test),2))
