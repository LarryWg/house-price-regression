import streamlit as st
import pandas as pd
import boto3, os #AWS SDK for Python
import requests
import plotly.express as px
from pathlib import Path



API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")
S3_BUCKET = os.getenv("S3_BUCKET", "larry-house-price-regression-data")
REGION = os.getenv("AWS_REGION", "ca-central-1")

s3 = boto3.client("s3", region_name=REGION)

def load_data_from_s3(key, local_path):
    """Download from s3 if not cached locally."""

    local_path = Path(local_path)
    if not local_path.exists():
        os.makedirs(local_path.parent, exist_ok=True)
        s3.download_file(S3_BUCKET, key, str(local_path))
        print(f"📥 Downloaded {key} from S3 to {local_path}")

    return str(local_path)

HOLDOUT_ENG_PATH = Path(load_data_from_s3("processed/holdout_engineered.csv", "data/processed/holdout_engineered.csv"))

HOLDOUT_CLEAN_PATH = Path(load_data_from_s3("processed/holdout_cleaned.csv", "data/processed/holdout_cleaned.csv"))




@st.cache_data
def load_holdout_data():
    fe = pd.read_csv(HOLDOUT_ENG_PATH)
    clean = pd.read_csv(HOLDOUT_CLEAN_PATH, parse_dates=["date"])[["date", "city_full"]]

    if len(fe) != len(clean):
        st.warning("Holdout datasets have different number of rows. Check data integrity.")
        min_len = min(len(fe), len(clean))
        fe = fe.iloc[:min_len].copy()
        clean = clean.iloc[:min_len].copy()
    
    display = pd.DataFrame(index = fe.index)
    display["date"] = clean["date"]
    display["region"] = clean["city_full"]
    display["year"] = display["date"].dt.year
    display["month"] = display["date"].dt.month
    display["actual_price"] = fe["price"]

    return fe, display


fe_df, display_df = load_holdout_data()


#### UI ####
st.title("House Price Regression ")

years = sorted(display_df["year"].unique())
months = list(range(1, 13))
regions = ["All"] + sorted(display_df["region"].dropna().unique())   

col1, col2, col3 = st.columns(3)
with col1:
    selected_year = st.selectbox("Select Year", options=years, index=0)
with col2:      
    selected_month = st.selectbox("Select Month", options=months, index=0)
with col3:
    selected_region = st.selectbox("Select Region", options=regions, index=0)


if st.button("Predict"):
    mask = (display_df["year"] == selected_year) & (display_df["month"] == selected_month)
    if selected_region != "All":
        mask &= (display_df["region"] == selected_region)

    index = display_df.index[mask]

    if len(index) == 0:
        st.warning("No data for selected filters.")
    else:
        st.write(f"Predicting for **{selected_year} - {selected_month:02d}** in **{selected_region}**")

        payload = fe_df.loc[index].to_dict(orient="records")
        try:
            response = requests.post(API_URL, json=payload, timeout=60)
            response.raise_for_status()
            out = response.json()
            preds = out.get("predictions", [])
            actuals = out.get("actual_price", None)

            view = display_df.loc[index, ["date", "region", "actual_price"]].copy()
            view = view.sort_values("date")
            view["prediction"] = pd.Series(preds, index=view.index).astype(float)

            mae = (view["prediction"] - view["actual_price"]).abs().mean()
            rmse = ((view["prediction"] - view["actual_price"]) ** 2).mean() ** 0.5
            avg_error_pct = ((view["prediction"] - view["actual_price"]).abs() / view["actual_price"]).mean() * 100

            st.subheader("Predictions vs Actuals")
            st.dataframe(view[["date", "region", "actual_price", "prediction"]].reset_index(drop=True), use_container_width=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("MAE", f"${mae:,.2f}")
            with c2:    
                st.metric("RMSE", f"${rmse:,.2f}")
            with c3:
                st.metric("Avg Error %", f"{avg_error_pct:.2f}%")


            """
            TREND CHART
            """

            if selected_region == "All":
                yearly_data = display_df[display_df["year"] == selected_year].copy()
                idx_all = yearly_data.index
                payload_all = fe_df.loc[idx_all].to_dict(orient="records")

                resp_all = requests.post(API_URL, json=payload_all, timeout=60)
                resp_all.raise_for_status()
                preds_all = resp_all.json().get("predictions", [])

                yearly_data["prediction"] = pd.Series(preds_all, index=yearly_data.index).astype(float)
            else:
                yearly_data = display_df[(display_df["year"] == selected_year) & (display_df["region"] == selected_region)].copy()
                idx_region = yearly_data.index
                payload_region = fe_df.loc[idx_region].to_dict(orient="records")

                resp_region = requests.post(API_URL, json=payload_region, timeout=60)
                resp_region.raise_for_status()
                preds_region = resp_region.json().get("predictions", [])

                yearly_data["prediction"] = pd.Series(preds_region, index=yearly_data.index).astype(float)

            monthly_avg = yearly_data.groupby("month")[["actual_price", "prediction"]].mean().reset_index()

            # Highlight selected month
            monthly_avg["highlight"] = monthly_avg["month"].apply(lambda m: "Selected" if m == selected_month else "Other")

            fig = px.line(
                monthly_avg,
                x="month",
                y=["actual_price", "prediction"],
                markers=True,
                labels={"value": "Price", "month": "Month"},
                title=f"Yearly Trend — {selected_year}{'' if selected_region=='All' else f' — {selected_region}'}"
            )

            # Add highlight with background shading
            highlight_month = selected_month
            fig.add_vrect(
                x0=highlight_month - 0.5,
                x1=highlight_month + 0.5,
                fillcolor="red",
                opacity=0.1,
                layer="below",
                line_width=0,
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error("Error during prediction. Check API and data integrity.")
            st.exception(e)
else:
    st.info("Choose filter and click **Predict**")
