import streamlit as st
import pandas as pd

df = pd.read_csv("Superstore.csv")

st.title("Superstore Dashboard")

st.write(df.head())

st.bar_chart(df.groupby("Category")["Sales"].sum())
