# train_model.py
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

data = 'normal_Anormal_dataset.csv'
model_path = 'lcr_model.pkl'

# خوا  ندن داده
df = pd.read_csv(data, header=None)
X = df.iloc[:, 0:17]
y = df.iloc[:, 17]

# تقسیم آموزش/تست
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# آموزش مدل
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# ارزیابی
y_pred = model.predict(X_test)


# ذخیره مدل
joblib.dump(model, model_path)

