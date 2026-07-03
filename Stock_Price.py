import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression #ml 
from sklearn.metrics import r2_score #for accuracy check


st.set_page_config(page_title="Stock Price Prediction",layout="centered")

st.write("📈 Stock Price Prediction System using Machine Learning")

# data load

df=pd.read_csv("StockPrice.csv")

with st.expander("📜 View Dataset"):
    st.dataframe(df)

# input data and output
x=df[["Open","High","Low","Volume"]]
y=df["Close"]


# device data for train test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=42)


# train the selected model

model=LinearRegression()
model.fit(x_train,y_train)

# model accurecy

y_pred = model.predict(x_test)

score=r2_score(y_test,y_pred)

with st.container(border=True):
    st.subheader("Predict future Stock Price 🤑?")

    Open =st.number_input("Enter your open price :", min_value=100,max_value=500,step=100)
    High =st.number_input("Enter your high price:",min_value=100,max_value=500,step=1)
    Low =st.number_input("Enter your low price :",min_value=100,max_value=500,step=1)
    Valume=st.number_input("Enter your volume:",min_value=100000,max_value=500000,step=1000)


    if st.button("Predict Price 📈"):
        prediction = model.predict([[Open,High,Low,Valume]])
        


        st.success(f"Predicted stock Price:₹{prediction[0]:,.2f}")
        st.info(f"Model Accurecy (R2 score) :{score:4f}")

        st.subheader("📊 open price vs close Price")

        fig,ax=plt.subplots(figsize=(6,3))

        ax.scatter(df["Open"],df["Close"],color="blue",s=50,label="Stock Data")

        ax.plot(df["Open"],model.predict(x),color="red",linewidth=2,label="Regression Line")
        ax.set_title("Open price vs Close Price")
        ax.set_xlabel("Open price")
        ax.set_ylabel(" Close price")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)