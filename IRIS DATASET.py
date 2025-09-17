import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# === 1) Cargar datos y entrenar modelo ===
iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# === 2) Matriz de confusión ===
cm = confusion_matrix(y_test, y_pred)
labels = iris.target_names

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Verdadero")
plt.tight_layout()
plt.show()

# === 3) Gráfico del reporte de clasificación ===
# Convertimos el reporte a diccionario para graficar
report_dict = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose().iloc[:-3, :]  # quitar accuracy y promedios

# Graficar precision, recall, f1-score por clase
report_df[['precision','recall','f1-score']].plot(kind='bar', figsize=(8,5))
plt.title("Reporte de Clasificación")
plt.ylabel("Score")
plt.ylim(0, 1.1)
plt.legend(loc='lower right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
