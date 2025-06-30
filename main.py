from plot_module import plot_data
from analysis_module import predict
import pandas as pd

def main():
    try:
        df = pd.read_csv("lcr.csv")
    except Exception as e2:
        print("Error in reading CSV file:", e2)
        return

    for col in df.columns[1:]:
        df[col] = df[col].astype(str).str.replace(",", "").astype(float)

    time_cols = [f"hist{i}" for i in range(1, 13)] + ["Present_value"]
    lcr_row = df[df["Title"] == "LCR"]
    if lcr_row.empty:
        raise ValueError(" LCR row not found in the dataset!")

    lcr_values = lcr_row.iloc[0][time_cols].astype(float).tolist()

    # بررسی آنومالی
    is_anomaly = predict(lcr_values)


    # رسم نمودار با درنظر گرفتن آنومالی
    plot_data(df, lcr_values, is_anomaly=is_anomaly)

if __name__ == "__main__":
    main()

