import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
df = pd.read_csv('bank-additional-full.csv', sep=';')
categorical_features = df.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop(columns=['y'])
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results['Predicted'] = results['Predicted'].apply(lambda x: 'will purchase' if x == 1 else 'will not purchase')
print(results.head(10))  # Display the first 10 predictions
purchase_counts = results['Predicted'].value_counts()
plt.figure(figsize=(8, 6))
purchase_counts.plot(kind='bar', color=['blue', 'orange'])
plt.title('Purchase Predictions')
plt.xlabel('Prediction')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()
