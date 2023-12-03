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

# Load background image
background_image = Image.open('Sleep Website Background.jpeg')

# Set background image using CSS
st.markdown(
    f"""
    <style>
        body {{
            background-image: url('{background_image}');
            background-size: cover;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Image
image_sleep = Image.open('sleep.png')
st.image(image_sleep, width=500, use_column_width=True)

# Title
st.title("Sleep Quality Prediction")

# Sidebar header and a separator
st.sidebar.header("Dashboard")
st.sidebar.markdown("---")

#get model
model_mode = st.sidebar.selectbox('🔎 Select Model',['Linear Regression','Random Forest'])
    
# Dropdown menu for selecting the page mode (Introduction, Visualization, Prediction, Deployment)
app_mode = st.sidebar.selectbox('🔎 Select Page',['Introduction','Visualization','Prediction','Deployment','Conclusion'])

# Dropdown menu for selecting the dataset (currently only "Salary" is available)
select_dataset =  st.sidebar.selectbox('💾 Select Dataset',["SleepHealth"])

# Load the salary quality dataset
df_sleep = pd.read_csv("SleepHealth.csv")
#####################################################################
# Changes made to data
df = df_sleep.dropna()

#########
# Create new df just in case
df = df.drop(['Person ID', 'Blood Pressure'], axis =1)
df2 = df.copy()
#########


#####################################################################


# Dropdown menu for selecting which variable from the dataset to predict
list_var = df2.columns
select_variable =  st.sidebar.selectbox('🎯 Select Variable to Predict',['Quality of Sleep'])

#  page content
if app_mode == 'Introduction':
    
    st.info("The dataset contains data on factors that affect sleep health.")
    st.info("This website will be able to predict your quality of sleep on a scale of 0 to 10.")
   
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
        width: 135px;
        height: 205px;
        font-size: 15px;
    """

    # Apply to each column
    col1,col2,col3,col4,col5 = st.columns(5)
    col1.markdown(f'<div style="{box_style}"><strong>Gender</strong><br><br><br>The gender of the person (Male/Female)</div>', unsafe_allow_html=True)
    col2.markdown(f'<div style="{box_style}"><strong>Age</strong><br><br><br>The age of the person in years</div>', unsafe_allow_html=True)
    col3.markdown(f'<div style="{box_style}"><strong>Sleep Duration</strong><br><br><br>The number of hours spent sleeping in a day</div>', unsafe_allow_html=True)
    col4.markdown(f'<div style="{box_style}"><strong>Quality of Sleep</strong><br><br><br>Subjective rating of sleep quality (0-10)</div>', unsafe_allow_html=True)
    col5.markdown(f'<div style="{box_style}"><strong>Physical Activity Level</strong><br><br>Number of minutes spent exercising in a day</div>', unsafe_allow_html=True)

    col6,col7,col8,col9,col10 = st.columns(5)
    col6.markdown(f'<div style="{box_style}"><strong>Stress Level</strong><br><br><br>Subjective rating of stress level experienced</div>', unsafe_allow_html=True)
    col7.markdown(f'<div style="{box_style}"><strong>BMI Category</strong><br><br><br>Whether a person is underweight, normal, or overweight</div>', unsafe_allow_html=True)
    col8.markdown(f'<div style="{box_style}"><strong>Heart Rate</strong><br><br><br>Resting heart rate in beats per minute</div>', unsafe_allow_html=True)
    col9.markdown(f'<div style="{box_style}"><strong>Daily Steps</strong><br><br><br>Number of steps walked in a day</div>', unsafe_allow_html=True)
    col10.markdown(f'<div style="{box_style}"><strong>Sleep Disorder</strong><br><br><br>Whether a person has a sleep disorder or not</div>', unsafe_allow_html=True)
##############
    st.markdown("")


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
    completeness = round((nonmissing.sum()/11)/len(df2), 2)
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
    #tab1, tab2 = st.tabs(["Line Chart", "📈 Correlation"])

    # Content for the "Line Chart" tab
    #tab1.subheader("Line Chart")
    # Display a line chart for the selected variables

    
    st.write(symbols)

    
    st.line_chart(data=df2, x=symbols[0], y=symbols[1], width=0, height=0, use_container_width=True)
    # Display a bar chart for the selected variables
    st.bar_chart(data=df2, x=symbols[0], y=symbols[1], use_container_width=True)

    # Content for the "Correlation" tab   

        fig7, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot each categorical variable
    sns.countplot(ax=axes[0], data=df, x="Gender")
    sns.countplot(ax=axes[1], data=df, x="BMI Category")
    sns.countplot(ax=axes[2], data=df, x="Sleep Disorder")

    # Set titles for each subplot
    axes[0].set_title("Gender")
    axes[1].set_title("BMI Category")
    axes[2].set_title("Sleep Disorder")

    # Adjust layout
    plt.tight_layout()

    # Show the plots
    st.pyplot(fig7)

    
    #tab2.subheader("Correlation Tab 📉")
    no_cat_df = df2.drop(['Gender', 'BMI Category', 'Occupation', 'Sleep Disorder'], axis = 1)
    # Create a heatmap to show correlations between variables in the dataset
    #fig, ax = plt.subplots(figsize=(width1, width1))
    
    #fig = sns.heatmap(no_cat_df.corr())
# , cmap=sns.cubehelix_palette(8), annot=True, ax=ax
    #st.pyplot(fig)
    #tab2.write(fig)

    plot = sns.heatmap(no_cat_df.corr(), cmap=sns.cubehelix_palette(8), annot=True)
 
    # Display the plot in Streamlit
    st.pyplot(plot.get_figure())
    #tab2.write(fig)
    # Display a pairplot for the first five variables in the dataset
    st.markdown("### Pairplot")
    df3 = df2
    fig3 = sns.pairplot(df2)
    st.pyplot(fig3)
    
    # fig4 = sns.histplot(data=df, x="Gender")
    # st.pyplot(fig4.get_figure())
    
    # fig5 = sns.histplot(df2["BMI Category"])
    # st.pyplot(fig5)
    
    # fig6 = sns.histplot(df2["Sleep Disorder"])
    # st.pyplot(fig6)

    ####FEATURE ENGINEERING FREQUENCY GRAPHS####
    # Select categorical variables with small number unique values
    #df_barplot = dataset[["Gender","BMI Category"]]

    # Create barplot with frequency for each variable
    #plt.figure(figsize=(8,8))

    #for c,var in enumerate(df_barplot.columns):
      # compute frequency of each unique value
      #df = df_barplot[var].value_counts(normalize=True).to_frame("frequency").reset_index()
      #df["frequency"] = df["frequency"]*100

      # plot the barplot
      #plt.subplot(3,2,c+1)
      #sns.barplot(data=df, x="index", y="frequency")
      #plt.title(str(var))
      #plt.xlabel("")
      #plt.ylabel("")

    #plt.tight_layout()


# Check if the app mode is set to 'Prediction'
if app_mode == 'Prediction':
    # Display a header for the Prediction section
    st.markdown("## Prediction")

    # Allow users to adjust the size of the training dataset using a slider in the sidebar
    test_size = st.sidebar.number_input("Train Set Size", min_value=0.00, step=0.01, max_value=1.00, value=0.20)
    # Drop the selected variable from the dataset to prepare for prediction
    #pred_df = df2.drop(labels=select_variable, axis=1)
    pred_df = df2
    #list_var = pred_df.columns

    
    from sklearn import preprocessing
        
    label_encoder = preprocessing.LabelEncoder()
    pred_df['Gender'] = label_encoder.fit_transform(pred_df['Gender'])
    pred_df['Occupation'] = label_encoder.fit_transform(pred_df['Occupation'])
    pred_df['BMI Category'] = label_encoder.fit_transform(pred_df['BMI Category'])
    pred_df['Sleep Disorder'] = label_encoder.fit_transform(pred_df['Sleep Disorder'])
    
    # Allow users to select explanatory variables for prediction
    output_multi = st.multiselect("Select Explanatory Variables",  ['Physical Activity Level','Sleep Duration','Stress Level','BMI Category','Heart Rate','Daily Steps','Sleep Disorder', 'Gender'], default = ['Physical Activity Level','Sleep Duration','Stress Level','BMI Category','Heart Rate','Daily Steps','Sleep Disorder', 'Gender'])


    
    # Define a function to perform linear regression prediction
    def predict_ml(target_choice, test_size, df, output_multi):
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
        
        # from sklearn import preprocessing
        
        # label_encoder = preprocessing.LabelEncoder()
        # df['Gender'] = label_encoder.fit_transform(df['Gender'])
        # df['Occupation'] = label_encoder.fit_transform(df['Occupation'])
        # df['BMI Category'] = label_encoder.fit_transform(df['BMI Category'])
        # df['Sleep Disorder'] = label_encoder.fit_transform(df['Sleep Disorder'])
        # Select the explanatory variables based on user input
       ####################################################################################
        # X = df.drop(['Occupation', target_choice], axis = 1)
        df = df.drop(['Occupation'], axis=1)
        X = df[output_multi]
        y = df[target_choice]

    
        ####################################################################################
        #new_df2 = new_df[output_multi]
       # x = new_df2
        #y = df[target_choice]
    
        # Display the top 25 rows of the explanatory and target variables in the Streamlit app
        # col1, col2 = st.columns(2)
        # col1.subheader("Feature Columns top 25")
        # col1.write(X.head(25))
        # col2.subheader("Target Column top 25")
        # col2.write(y.head(25))
    
        # Split the data into training and testing sets
        # X = pd.get_dummies(data=X, drop_first=True)
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = test_size,random_state=42)
    
        # Initialize and train a linear regression model
        if model_mode == "Linear Regression":
            
            lm = LinearRegression()
            lm.fit(X_train,y_train)
        
            # Predict the target variable for the test set
            predictions = lm.predict(X_test)
    
            return X_train, X_test, y_train, y_test, predictions, X, y
        else: 
            from sklearn.ensemble import RandomForestRegressor            
            rf = RandomForestRegressor()
            rf.fit(X_train,y_train)
        
            # Predict the target variable for the test set
            predictions = rf.predict(X_test)
    
            return X_train, X_test, y_train, y_test, predictions, X, y

        # Check if the DataFrame is not empty
    if pred_df.empty or len(output_multi) == 0 or select_variable not in pred_df.columns:
        st.warning("Please select at least one variable for prediction.")
    else:
        # Call the prediction function and store the results
        X_train, X_test, y_train, y_test, predictions, X, y = predict_ml(select_variable, test_size, pred_df, output_multi)

        # Display the results header in the Streamlit app
        st.subheader('🎯 Results')

        # Display prediction metrics
        st.write("1) The model explains,", np.round(mt.explained_variance_score(y_test, predictions) * 100, 2),
                 "% variance of the target feature")
        st.write("2) The Mean Absolute Error of the model is:", np.round(mt.mean_absolute_error(predictions, y_test), 2))
        st.write("3) MSE: ", np.round(mt.mean_squared_error(predictions, y_test), 2))
        st.write("4) The R-Square score of the model is ", np.round(mt.r2_score(predictions, y_test), 2))

#######

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


if app_mode == 'Deployment':
    # Deployment page for model deployment
    st.markdown("# :violet[Deployment 🚀]")
    id = st.text_input('ID Model', '/content/mlruns/2/ab897d9145a64b6d87b0a54e1c52735c/artifacts/to_model_v1')

    # Load model for prediction
    logged_model = f'/content/mlruns/2/ab897d9145a64b6d87b0a54e1c52735c/artifacts/top_model_v1'
    loaded_model = mlflow.pyfunc.load_model(logged_model)



    df = pd.read_csv("SleepHealth.csv")
    deploy_df= df.drop(labels='alcohol', axis=1)
    list_var = deploy_df.columns
    #st.write(target_choice)

    number1 = st.number_input(deploy_df.columns[0],0.7)
    number2 = st.number_input(deploy_df.columns[1],0.04)
    number3 = st.number_input(deploy_df.columns[2],1.1)
    number4 = st.number_input(deploy_df.columns[3],0.05)
    number5 = st.number_input(deploy_df.columns[4],25)
    number6 = st.number_input(deploy_df.columns[5],20)
    number7 = st.number_input(deploy_df.columns[6],0.98)
    number8 = st.number_input(deploy_df.columns[7],1.9)
    number9 = st.number_input(deploy_df.columns[8],0.4)
    number10 = st.number_input(deploy_df.columns[9],9.4)
    number11 = st.number_input(deploy_df.columns[10],5)

    data_new = pd.DataFrame({deploy_df.columns[0]:[number1], deploy_df.columns[1]:[number2], deploy_df.columns[2]:[number3],
         deploy_df.columns[3]:[number4], deploy_df.columns[4]:[number5], deploy_df.columns[5]:[number6], deploy_df.columns[6]:[number7],
         deploy_df.columns[7]:[number8], deploy_df.columns[8]:[number9],deploy_df.columns[9]:[number10],deploy_df.columns[10]:[number11]})
    # Predict on a Pandas DataFrame.
    #import pandas as pd
    st.write("Prediction :", np.round(loaded_model.predict(data_new)[0],2))


if app_mode == 'Conclusion':
    # Display dataset details
    st.markdown("### Key Takeaways")
    st.image(
            "https://media.tenor.com/jQlbcSS2HgoAAAAd/tom-and-jerry-sleep.gif",
            width=500,
        )
    st.markdown("")
    st.markdown("")
    st.markdown("#### **About Model:**")
    st.info("The model showcases a strong capability to predict sleep quality level based on the selected variables.")
    st.info("The high R-squared value suggests that the model explains a substantial portion of the variability in sleep quality level.")
    st.info("Both the low Mean Absolute Error and Mean Squared Error demonstrate that the model's predictions are accurate and have minimal errors.")
    st.markdown("")
    st.markdown("#### **About Sleep Quality:**")
    st.info("The higher your sleep duration, the bigger the positive impact on level of sleep quality.")
    st.write("Increase sleep duration for a greater chance at having a better sleep quality.")
    st.info("The higher your stress level, the bigger the negative impact on level of sleep quality.")
    st.write("Decrease stress level for a greater chance at having a better sleep quality.")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.image(
            "https://sapro.moderncampus.com/hs-fs/hubfs/Destiny/Imported_Blog_Media/source-Apr-05-2022-02-33-39-07-PM.gif?width=904&height=530&name=source-Apr-05-2022-02-33-39-07-PM.gif",
            width=400,
        )
