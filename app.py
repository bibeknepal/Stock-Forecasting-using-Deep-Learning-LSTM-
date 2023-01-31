import numpy as np # Fundamental package for scientific computing with Python
import pandas as pd # Additional functions for analysing and manipulating data
import plotly.graph_objs as go
import matplotlib.pyplot as plt # Important package for visualization 
import streamlit as st
from sklearn.preprocessing import MinMaxScaler # This Scaler removes the median and scales the data according to the quantile range to normalize the price data 
# from tensorflow.keras.models import load_model
from keras.models import load_model

isLoaded = False

st.title("Stock Forecasting Web Application")

stocks = ("NABIL","ADBL","HIDCL","NLIC","UNL","UPPER")


def onSelectedStockChange():
    st.session_state.clear() 

selected_stock = st.selectbox("Select Dataset for Prediction", stocks, on_change=onSelectedStockChange)

#Load the data
@st.cache
def load_data(ticker):
    df = pd.read_csv(f"data\\{ticker}.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Date"] = df["Date"].dt.date
    df.index=df["Date"]
    df.drop("Date",axis=1,inplace=True)
    return df

if(st.button("Load Data")):
    isLoaded = True
    st.session_state.setdefault("loaded", isLoaded)    

   
if st.session_state.get("loaded"):   
    df = load_data(selected_stock)
    data_load_state = st.text("Loading data...")
    # df = load_data(selected_stock)
    data_load_state.text("Data Loaded!")
    st.subheader("Loaded Data")
    
    #Plot the Loaded Data
    def Plot_loaded_data():
        fig = go.Figure()
        # fig.add_trace(go.Scatter(x = df.index, y = df['Open'], name = 'Open Price'))
        fig.add_trace(go.Scatter(x = df.index, y = df['Close'], name = 'Close Price', line = dict(color = 'blue', width = 2)))
        fig.layout.update(title_text = "Graph Of Close Price", hovermode = "x")
        st.plotly_chart(fig)

    Plot_loaded_data()

    #Filter the columns needed
    def filter_data(df):
        # Indexing Batches
        test_df = df.sort_values(by=['Date']).copy()

        # List of considered Features
        FEATURES = ['Volume', 'High', 'Low', 'Open', 'Close'
                #, 'Month', 'Year', 'Adj Close'
                ]
        # Create the dataset with features and filter the data to the list of FEATURES
        data = pd.DataFrame(test_df)
        data_filtered = data[FEATURES]
        return data_filtered
    data_filtered = filter_data(df)

    #Preprocessing Data
    def preprocess_data(data):
        # Get the number of rows in the data
        nrows = data.shape[0]

        # Convert the data to numpy values
        np_data_unscaled = np.array(data)
        np_data = np.reshape(np_data_unscaled, (nrows, -1))
        

        # Transform the data by scaling each feature to a range between 0 and 1
        scaler = MinMaxScaler()
        np_data_scaled = scaler.fit_transform(np_data_unscaled)

        return np_data_scaled, scaler
    data_scaled, scaler = preprocess_data(data_filtered)


    sequence_length = 100
    n_steps = sequence_length
    data_took_as_input = data_filtered[-n_steps:]
    st.subheader("Data took as input")
    st.write(data_took_as_input)


    x_input = data_scaled[-n_steps:, :]
    x_input = x_input.reshape((1, n_steps, len(data_filtered.axes[1])))

    model = load_model("model\\my_model.h5")

    forecast = st.slider("Choose the number of future days to predict", 1 , 30)

    if st.button("Predict"):
        # Initialize an empty list to store the predictions
        lst_output = []


        for i in range(forecast):
            # Make a prediction
            yhat = model.predict(x_input, verbose=0)
            # Append the prediction to the list
            lst_output.append(yhat[0])
            yhat = yhat.reshape((1, 1, yhat.shape[1]))
            # Update x_input with the last 100 input and the previous prediction
            x_input = np.concatenate((x_input[:,1:,:], yhat), axis = 1)

        lst_output = scaler.inverse_transform(lst_output)
        lst_output_close = lst_output[:,-1]

        dataset = data_filtered.filter(['Close'])
        # Use pandas to concatenate the forecasted values to the last of the dataset
        forecasted_values = pd.DataFrame(lst_output_close, columns=['Close'])
        original_data_length = len(dataset)
        index = pd.date_range(start=dataset.index[-1] + pd.DateOffset(1), periods=forecast, freq="D")
        forecasted_values.set_index(index,inplace=True)
        st.subheader(f"Forecasting future {forecast} days close price")
        st.write(forecasted_values)
        dataset = pd.concat([dataset, forecasted_values], axis=0)

        # Create a trace for the original close price data
        original_trace = go.Scatter(x=dataset.index[-sequence_length-forecast:-forecast],
                                    y=dataset['Close'][-sequence_length-forecast:-forecast],
                                    mode='lines',
                                    name='Original')

        # Create a trace for the forecasted close price data
        forecasted_trace = go.Scatter(x=dataset.index[-forecast:],
                                y=dataset['Close'][-forecast:],
                                mode='lines',
                                name='Forecasted Close',
                                line=dict(color='green'))

        # Create the plotly figure
        fig = go.Figure(data=[original_trace, forecasted_trace],
                    layout=go.Layout(title='Forecasted Graph',
                                    xaxis_title='Date',
                                    yaxis_title='Close Price'))

        # Display the plotly figure in Streamlit
        st.plotly_chart(fig)

else:
    st.write("Click Load Data to load the data")