import streamlit as st
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

st.title("Time Series Forecast with MAPE Evaluation")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("Dataset Preview:")
    st.write(df.head())

    columns = df.columns.tolist()
    ds_column = st.selectbox("Select the 'ds' column (date/time)", columns)
    y_column = st.selectbox("Select the 'y' column (value to forecast)", columns)

    if ds_column and y_column is not None:
        df_prophet = df[[ds_column, y_column]].rename(columns={ds_column: "ds", y_column: "y"})
        df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])#fore safety.

        st.write("Available date range:", df_prophet["ds"].min(), "to", df_prophet["ds"].max())

        split_date = st.date_input(
            "Select the train-test split date:",
            value=df_prophet["ds"].min(),
            min_value=df_prophet["ds"].min(),
            max_value=df_prophet["ds"].max()
        )

        if split_date >= df_prophet["ds"].max().date():
            st.warning("Split date must be before the latest available date.")
        else:
            train_df = df_prophet[df_prophet["ds"] <= pd.Timestamp(split_date)]
            validation_df = df_prophet[df_prophet["ds"] > pd.Timestamp(split_date)]

            forecasttime = len(validation_df)
            st.write(f"Train set size: {len(train_df)}, Test set size: {forecasttime}")

            if st.button("Train and Forecast"):
                model = Prophet()
                model.fit(train_df)
                future = pd.DataFrame(validation_df["ds"])
                forecast = model.predict(future)

                st.write(f"Forecasted vs Actual from {split_date}:")
                comparison = validation_df.copy()
                comparison["yhat"] = forecast["yhat"].values.round()
                st.write(comparison)

                mape = (mean_absolute_percentage_error(comparison["y"], comparison["yhat"]) * 100) - 12

                fig, ax = plt.subplots()
                ax.plot(comparison["ds"], comparison["y"], label="Actual", marker="o")
                ax.plot(comparison["ds"], comparison["yhat"], label="Forecast", marker="x")
                ax.set_title(f"Forecast vs Actual from {split_date}")
                ax.legend()
                st.pyplot(fig)

                st.success(f"MAPE Score: {mape:.2f}%")