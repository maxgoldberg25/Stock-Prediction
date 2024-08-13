# Stock Price Prediction using LSTM

This repository contains a comprehensive implementation of an Artificial Recurrent Neural Network, specifically a Long Short-Term Memory (LSTM) network, designed to predict the stock price of Tesla, Inc. (TSLA). The model leverages historical stock data over a 60-day period to forecast future prices, employing Python and Jupyter Notebook as the core development tools.


## Introduction

Predicting stock prices is inherently complex due to the stochastic nature of financial markets. This project utilizes a Long Short-Term Memory (LSTM) network, a type of Artificial Recurrent Neural Network (RNN), to model sequential dependencies within stock prices. The aim is to deliver accurate predictions of Tesla's stock price based on the past 60 days of trading data.

## Dataset

The dataset employed consists of historical stock prices for Tesla (TSLA), including the following features:

- Date
- Open
- High
- Low
- Close
- Volume

The data can be sourced from reliable financial data providers such as Yahoo Finance, Alpha Vantage, or other equivalent platforms.

## Installation

Clone this repository to your local machine and install the required Python packages:

```bash
git clone git clone git clone https://github.com/maxgoldberg25/Stock-Prediction.git
cd Stock_Price_Prediction_Model_(Python_3).ipynb
```

### Prerequisites

Ensure that the following libraries are installed:

- **Python 3.x:** The programming language used to build the model.
- **TensorFlow / Keras:** For building and training the LSTM model.
- **NumPy:** To handle numerical operations.
- **Pandas:** For data manipulation and analysis.
- **Matplotlib:** For plotting and visualizing the results.
- **Scikit-learn:** For data preprocessing.
- **Jupyter Notebook:** To run the notebook and interact with the code.

You can install the required packages using pip:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn jupyter
```

## Follow the Notebook's Instructions

The notebook contains the following steps:

### Data Preprocessing
- **Import required libraries**, including `pandas_datareader`, `NumPy`, `Pandas`, `MinMaxScaler`, `LSTM`, and `Matplotlib`.
- **Load the historical stock data** for Tesla (TSLA) using `pandas_datareader` from Yahoo Finance.
- **Visualize the closing price history** to understand the trends.
- **Filter the 'Close' column** and scale the data using `MinMaxScaler`.
- **Create the training and test datasets** with an 80-20 split and generate sequences of 60-day historical data.

### Model Construction
- **Build the LSTM model** using the Sequential API from Keras.
- The model includes:
  - An **LSTM layer** with 50 units, returning sequences.
  - Another **LSTM layer** with 50 units, not returning sequences.
  - A **Dense layer** with 25 units.
  - A **final Dense layer** with 1 unit for the output.

### Model Training
- **Compile the model** using the Adam optimizer and mean squared error as the loss function.
- **Train the model** on the training data with a batch size of 1 and 1 epoch.
  - The training process completed with a loss of approximately **3.30e-04**.

### Prediction & Visualization
- **Create the test dataset** using the last 60 days of the training data and the test data.
- **Make predictions** using the trained LSTM model.
- **Inverse transform the predictions** back to the original price scale.
- **Calculate the RMSE** to evaluate the model's accuracy.
  - The calculated RMSE for the model is approximately **2.84**.
- **Visualize the predicted vs. actual stock prices** using Matplotlib.
  - The resulting plot shows the comparison between the actual stock prices and the predicted prices by the model, highlighting its predictive accuracy.

### Model Architecture
The LSTM model architecture includes the following components:

- **LSTM Layer**: Captures temporal dependencies and trends in the stock price sequence.
- **Dense Layer**: Outputs the final predicted stock price.

The architecture is optimized for time-series forecasting, ensuring the model's robustness in predicting future price movements.

### Results
After training, the model’s predictions are compared with the actual stock prices. Visualization through Matplotlib illustrates the model's performance:

![ActualVSPredicted](https://github.com/user-attachments/assets/c1aee156-f81d-43cd-b46b-4478dffe1524)


This graph highlights the model's ability to track and predict stock price movements with reasonable accuracy. The **Root Mean Square Error (RMSE)** of the predictions is approximately **2.84**.
