
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
import os
import imageio


# Image
image_sleep = Image.open('sleep.png')
st.image(image_sleep, width=500, use_column_width=True)

# Title
st.title("Sleep Health Prediction")

# Sidebar header and a separator
st.sidebar.header("Dashboard")
st.sidebar.markdown("---")

#get model
model_mode = st.sidebar.selectbox('🔎 Select Model',['Linear Regression','Logistic Regression'])
    
# Dropdown menu for selecting the page mode (Introduction, Visualization, Prediction, Deployment)
app_mode = st.sidebar.selectbox('🔎 Select Page',['Introduction','Visualization','Prediction','Deployment','Conclusion'])

# Dropdown menu for selecting the dataset (currently only "Salary" is available)
select_dataset =  st.sidebar.selectbox('💾 Select Dataset',["SleepHealth"])

# Load the salary quality dataset
df_sleep = pd.read_csv("SleepHealth.csv")
#####################################################################
# Changes made to data
df = df_sleep.dropna()
df = pd.concat([df, df['Blood Pressure'].str.split('/', expand=True)], axis=1).drop('Blood Pressure', axis=1)
df = df.rename(columns={0: 'BloodPressure_Upper_Value', 1: 'BloodPressure_Lower_Value'})
df['BloodPressure_Upper_Value'] = df['BloodPressure_Upper_Value'].astype(float)
df['BloodPressure_Lower_Value'] = df['BloodPressure_Lower_Value'].astype(float)
#########
# Create new df just in case
df2 = df.drop(['Person ID'], axis =1)
#########


#####################################################################






# Dropdown menu for selecting which variable from the dataset to predict
list_var = df2.columns
select_variable =  st.sidebar.selectbox('🎯 Select Variable to Predict',['Quality of Sleep'])

# Introduction page content
if app_mode == 'Introduction':
    
    st.info("The dataset contains data on factors that affect sleep health.")
    st.info("This website will be able to predict - ADD TEXTT blah blah blah.")
   
    # Display dataset details
    st.markdown("### 00 - Show Dataset")

#############
    # Split the page into 6 columns to display information about each salary variable
    ### UPDATE TO 12 COLUMNS
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    # CSS to style the columns
    box_style = """
        border: 2px solid #FFFFFF;
        padding: 10px; 
        margin: 5px;
        border-radius: 10px; 
        width: 110px;
        height: 250px;
        font-size: 15px;
    """

    # Apply to each column
    col1,col2,col3,col4,col5,col6 = st.columns(6)
    col1.markdown(f'<div style="{box_style}"><strong>Gender</strong><br><br><br>The gender of the person (Male/Female)</div>', unsafe_allow_html=True)
    col2.markdown(f'<div style="{box_style}"><strong>Age</strong><br><br><br>The age of the person in years.</div>', unsafe_allow_html=True)
    col3.markdown(f'<div style="{box_style}"><strong>Sleep Duration</strong><br><br>The number of hours spent sleeping in a day.</div>', unsafe_allow_html=True)
    col4.markdown(f'<div style="{box_style}"><strong>Quality of Sleep</strong><br><br><br>Subjective rating of sleep quality.</div>', unsafe_allow_html=True)
    col5.markdown(f'<div style="{box_style}"><strong>Physical Activity Level</strong><br><br>Number of minutes spent exercising in a day.</div>', unsafe_allow_html=True)
    col6.markdown(f'<div style="{box_style}"><strong>Stress Level</strong><br><br><br>Subjective rating of stress level experienced.</div>', unsafe_allow_html=True)

    col7,col8,col9,col10,col11,col12 = st.columns(6)
    col7.markdown(f'<div style="{box_style}"><strong>BMI Category</strong><br><br><br>Whether a person is underweight, normal, or overweight.</div>', unsafe_allow_html=True)
    col8.markdown(f'<div style="{box_style}"><strong>Blood Pressure Upper Value</strong><br><br><br>Pressure in arteries when the heart beats.</div>', unsafe_allow_html=True)
    col9.markdown(f'<div style="{box_style}"><strong>Blood Pressure Lower Value</strong><br><br><br>Pressure in arteries when the heart rests.</div>', unsafe_allow_html=True)
    col10.markdown(f'<div style="{box_style}"><strong>Heart Rate</strong><br><br>Resting heart rate in beats per minute.</div>', unsafe_allow_html=True)
    col11.markdown(f'<div style="{box_style}"><strong>Daily Steps</strong><br><br><br>Number of steps walked in a day.</div>', unsafe_allow_html=True)
    col12.markdown(f'<div style="{box_style}"><strong>Sleep Disorder</strong><br><br>Whether a person has a sleep disorder or not.</div>', unsafe_allow_html=True)
##############



    # Allow users to view either the top or bottom rows of the dataset
    num = st.number_input('No. of Rows', 5, 10)
    head = st.radio('View from top (head) or bottom (tail)', ('Head', 'Tail'))
    if head == 'Head':
        st.dataframe(df2.head(num))
    else:
        st.dataframe(df2.tail(num))

    # Display the shape (number of rows and columns) of the dataset
    st.markdown("Number of rows and columns helps us to determine how large the dataset is.")
    st.text('(Rows,Columns)')
    st.write(df2.shape)


    st.markdown("### 01 - Description")
    st.dataframe(df2.describe())



    st.markdown("### 02 - Missing Values")
    st.markdown("Missing values are known as null or NaN values. Missing data tends to **introduce bias that leads to misleading results.**")
    dfnull = df2.isnull().sum()/len(df2)*100
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
    nonmissing = (df2.notnull().sum().round(2))
    completeness= round(sum(nonmissing)/len(df2),2)
    st.write("Completeness ratio:",completeness)
    st.write(nonmissing)
    if completeness >= 0.80:
        st.success("Looks good! as we have completeness ratio greater than 0.85.")

    else:
        st.success("Poor data quality due to low completeness ratio( less than 0.85).")

############################################################################################################################################################################## STILL NEED TO EDIT FOR CURRENNT DF AND MODEL
if app_mode == 'Visualization':

    list_var = df2.columns
    # Display a header for the Visualization section
    st.markdown("## Visualization")

    # Allow users to select two variables from the dataset for visualization
    symbols = st.multiselect("Select two variables", list_var, ['Stress Level','Heart Rate'])

    # Create a slider in the sidebar for users to adjust the plot width
    width1 = st.sidebar.slider("plot width", 1, 25, 10)

    # Create tabs for different types of visualizations
    tab1, tab2 = st.tabs(["Line Chart", "📈 Correlation"])

    # Content for the "Line Chart" tab
    tab1.subheader("Line Chart")
    # Display a line chart for the selected variables

    
    st.write(symbols)

    
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
    output_multi = st.multiselect("Select Explanatory Variables",  ['Physical Activity Level','Sleep Duration','Stress Level','BMI Category','Heart Rate','Daily Steps','Sleep Disorder','Blood Pressure'])

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
        X = df[['Physical Activity Level','Sleep Duration','Stress Level','BMI Category','Heart Rate','Daily Steps','Sleep Disorder','Blood Pressure']]
        
        
        y = df['Quality of Sleep']
    
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



if app_mode == 'Chatbot 🤖':
    st.markdown("# :violet[ Your Personal Chatbot 🤖]")
   # OPENAI_API_KEY = "YOUR_API_KEY"
    # Set org ID and API key
    openai.organization = st.secrets.op_ai.org_key
    openai.api_key = st.secrets.op_ai.api_key

    # Initialise session state variables
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
    if 'model_name' not in st.session_state:
        st.session_state['model_name'] = []
    if 'cost' not in st.session_state:
        st.session_state['cost'] = []
    if 'total_tokens' not in st.session_state:
        st.session_state['total_tokens'] = []
    if 'total_cost' not in st.session_state:
        st.session_state['total_cost'] = 0.0

    # Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
    model_name = "GPT-3.5"
    counter_placeholder = st.sidebar.empty()
    counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
    clear_button = st.sidebar.button("Clear Conversation", key="clear")

    # Map model names to OpenAI model IDs
    if model_name == "GPT-3.5":
        model = "gpt-3.5-turbo"
    else:
        model = "gpt-4"

    # reset everything
    if clear_button:
        st.session_state['generated'] = []
        st.session_state['past'] = []
        st.session_state['messages'] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        st.session_state['number_tokens'] = []
        st.session_state['model_name'] = []
        st.session_state['cost'] = []
        st.session_state['total_cost'] = 0.0
        st.session_state['total_tokens'] = []
        counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")


    # generate a response
    def generate_response(prompt):
        st.session_state['messages'].append({"role": "user", "content": prompt})

        completion = openai.ChatCompletion.create(
            model=model,
            messages=st.session_state['messages']
        )
        response = completion.choices[0].message.content
        st.session_state['messages'].append({"role": "assistant", "content": response})

        # print(st.session_state['messages'])
        total_tokens = completion.usage.total_tokens
        prompt_tokens = completion.usage.prompt_tokens
        completion_tokens = completion.usage.completion_tokens
        return response, total_tokens, prompt_tokens, completion_tokens


    # container for chat history
    response_container = st.container()
    # container for text box
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_area("You:", key='input', height=100)
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output, total_tokens, prompt_tokens, completion_tokens = generate_response(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
            st.session_state['model_name'].append(model_name)
            st.session_state['total_tokens'].append(total_tokens)

            # from https://openai.com/pricing#language-models
            if model_name == "GPT-3.5":
                cost = total_tokens * 0.002 / 1000
            else:
                cost = (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000

            st.session_state['cost'].append(cost)
            st.session_state['total_cost'] += cost

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))
                st.write(
                    f"Model used: {st.session_state['model_name'][i]}; Number of tokens: {st.session_state['total_tokens'][i]}; Cost: ${st.session_state['cost'][i]:.5f}")
                counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")



if app_mode == 'Conclusion':
    # Display dataset details
    st.markdown("### Key Takeaways")
    st.image(
            "https://media.tenor.com/jQlbcSS2HgoAAAAd/tom-and-jerry-sleep.gif",
            width=400,
        )


