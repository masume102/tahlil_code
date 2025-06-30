
import matplotlib.pyplot as plt
import seaborn as sns

def plot_data(df, lcr_series, is_anomaly=False):
    time_cols = [f"hist{i}" for i in range(1, 13)] + ["Present_value"]

    fig, ax = plt.subplots(figsize=(14, 6))

    # رسم نقاط قبل از Present_value
    ax.plot(time_cols[:-1], lcr_series[:-1], label="LCR", color="blue", marker='o')

    # وصل کردن دو نقطه آخر (برای حفظ پیوستگی خط)
    ax.plot(time_cols[-2:], lcr_series[-2:], color="blue", marker='o')

    # اگر آنومالی است، نقطه آخر را با رنگ قرمز مشخص کن
    if is_anomaly:
        ax.scatter([time_cols[-1]], [lcr_series[-1]], color='red', label="Out of Appetite Point", marker='o', s=100)

    ax.set_xticklabels(time_cols, rotation=45)
    ax.set_title("LCR Trend")
    ax.set_xlabel("Time Period")
    ax.set_ylabel("LCR Value")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    # --- خروجی دوم: Heatmap درصد تأثیر ---
    df_items = df[df["Title"] != "LCR"].set_index("Title")
    df_items = df_items[time_cols]
    df_percent = df_items.div(df_items.sum(axis=0), axis=1).fillna(0) * 100

    plt.figure(figsize=(14, 10))
    sns.heatmap(df_percent, cmap="Reds", annot=True, fmt=".1f", linewidths=0.5)
    plt.title("Percentage Impact of Each Item on LCR Over Time")
    plt.xlabel("Time Points")
    plt.text(len(time_cols) + 0.5, len(df_percent)/2, "Weight (%)", rotation=90, verticalalignment='center', fontsize=12)
    plt.ylabel("Items")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

