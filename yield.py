import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from annoy import AnnoyIndex
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, PredefinedSplit
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('yield_df.csv', index_col=False)
df = df[['Area', 'Item', 'Year', 'hg/ha_yield', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']]
onehot_encoder = OneHotEncoder()
onehot_encoded = onehot_encoder.fit_transform(df[['Area']])
# Convert the one-hot encoded sparse matrix to a dense DataFrame
onehot_df = pd.DataFrame(onehot_encoded.toarray(), columns=onehot_encoder.get_feature_names_out(['Area']))
df = pd.concat([df, onehot_df], axis=1)

target = df['Item']
label_encoder = LabelEncoder()
encoded_target = label_encoder.fit_transform(target)
df['encoded_item'] = encoded_target
df.drop(columns=['Area', 'Item'], inplace=True)
X = df.drop(columns=['encoded_item'])
y = df['encoded_item']

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val)

print("Original dataset size:", X.shape[0])
print("Training set size:", X_train.shape[0])
print("Validation set size:", X_val.shape[0])
print("Test set size:", X_test.shape[0])
param_grid = {
'n_estimators' : [10, 20, 50, 100, 150, 200],
'max_depth' : [2, 4, 8, 16, 32],
'min_samples_split' : [2, 5, 10, 20],
'min_samples_leaf' : [2, 4, 8, 16, 32, 64]
}
X_pre_defined = np.concatenate((X_train, X_val), axis=0)
y_pre_defined = np.concatenate((y_train, y_val), axis=0)

split_index = [-1] * len(X_train) + [0] * len(X_val)
predefined_split = PredefinedSplit(test_fold = split_index)
model = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(model, param_grid, cv = predefined_split, scoring ='accuracy', verbose = 1)
grid_search.fit(X_pre_defined, y_pre_defined)

grid_search.fit(X_pre_defined, y_pre_defined)
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validated Accuracy:", grid_search.best_score_)
model = RandomForestClassifier(max_depth=32, min_samples_leaf=2, min_samples_split=5, n_estimators=200, random_state=42)
model = model.fit(X_train, y_train)

print("---Training---")
pred_train = model.predict(X_train)
print("Accuracy:", accuracy_score(y_train, pred_train) )
print("Precision:", precision_score(y_train, pred_train, average='weighted'))
print("Recall:", recall_score(y_train, pred_train, average='weighted'))
print("F1 Score:", f1_score(y_train, pred_train, average='weighted'))

print("---Validation---")
pred_val = model.predict(X_val)
print("Accuracy:", accuracy_score(y_val, pred_val))
print("Precision:", precision_score(y_val, pred_val, average='weighted'))
print("Recall:", recall_score(y_val, pred_val, average='weighted'))
print("F1 Score:", f1_score(y_val, pred_val, average='weighted'))
model = RandomForestClassifier(max_depth=32, min_samples_leaf=2, min_samples_split=5, n_estimators=200, random_state=42)

print("---Test---")
model = model.fit(X_train_val, y_train_val)
pred_test = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred_test) )
print("Precision:", precision_score(y_test, pred_test, average='weighted'))
print("Recall:", recall_score(y_test, pred_test, average='weighted'))
print("F1 Score:", f1_score(y_test, pred_test, average='weighted'))
cm = confusion_matrix(y_test, pred_test)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(len(cm)), yticklabels=range(len(cm)))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
df = pd.read_csv('Plant_Parameters.csv')
df.head()
features = df.drop(columns=['Plant Type']).to_numpy()
labels = df['Plant Type'].tolist()

f = features.shape[1]
index = AnnoyIndex(f, 'euclidean')

for i, vector in enumerate(features):
    index.add_item(i, vector)

index.build(10)
pH = 6
Soil_EC = 0.3
Phospohorus = 13
Potassium = 142
Urea = 38
TSP = 18
MOP = 26
Moisture = 74
Temp = 72
query_vector = [pH, Soil_EC, Phospohorus, Potassium, Urea, TSP, MOP, Moisture, Temp]

k = 3  # Number of nearest neighbors
nearest_neighbors = index.get_nns_by_vector(query_vector, n=k, include_distances=True)

from collections import Counter

neighbor_ids = nearest_neighbors[0]
neighbor_classes = [labels[i] for i in neighbor_ids]

predicted_class = Counter(neighbor_classes).most_common(1)[0][0]
print("Predicted Class:", predicted_class)