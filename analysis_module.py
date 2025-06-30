import numpy as np
import pandas as pd
import statistics
import joblib

def extract_features(new_series_input):
    """
    ورودی: لیستی از ۱۳ مقدار اولیه LCR شامل ۱۲ مقدار گذشته و 1 مقدار Present
    خروجی: لیست کامل شامل ۲۷ ویژگی استخراج‌شده (برای مدل)
    """
    remaining_features = [
        new_series_input[i+1] - new_series_input[i]
        for i in range(12)
    ]

    last_diff = abs(new_series_input[12] - new_series_input[11])
    previous_diffs_abs = [abs(x) for x in remaining_features[0:11]]
    median_prev = statistics.median(previous_diffs_abs)
    final_jump_ratio = last_diff / median_prev if median_prev != 0 else 0
    remaining_features.append(final_jump_ratio)

    # مجموع: 13 مقدار اولیه + 12 اختلاف + 1 نسبت پرش = 26 ویژگی
    return new_series_input + remaining_features

def predict(new_series_input, model_path='lcr_model.pkl'):
    """
    13 meqdare avalieh (12 jh feali + 1 gozashte) baray tashkhise anomaly
    """
    # بارگذاری مدل آموزش‌دیده
    model = joblib.load(model_path)

    # استخراج ویژگی‌ها
    features = extract_features(new_series_input)

    # تبدیل به DataFrame برای تطابق نام ستون‌ها با مدل
    features_df = pd.DataFrame([features])

    # پیش‌بینی
    prediction = model.predict(features_df)
    print("Anomaly?", bool(prediction[0]))

    return bool(prediction[0])
