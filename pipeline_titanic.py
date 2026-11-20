import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["family_size"] = X["sibsp"] + X["parch"] + 1
        X["is_alone"] = (X["family_size"] == 1).astype(int)
        return X

df = pd.read_csv("titanic.csv")
features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
X = df[features].copy()
y = df["survived"]

numeric_features = ["pclass", "age", "sibsp", "parch", "fare", "family_size", "is_alone"]
categorical_features = ["sex", "embarked"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])
pipeline = Pipeline([
    ("features", FeatureEngineering()),
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(random_state=42, class_weight="balanced"))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
param_grid = {
    "model__n_estimators": [150, 250],
    "model__max_depth": [None, 8, 12],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    pipeline, param_grid, cv=cv, scoring="accuracy", n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)

best_pipeline = grid_search.best_estimator_
predictions = best_pipeline.predict(X_test)
probabilities = best_pipeline.predict_proba(X_test)[:, 1]

print("Mejores parámetros:", grid_search.best_params_)
print("Mejor accuracy de validación cruzada:", grid_search.best_score_)
print("Accuracy de prueba:", accuracy_score(y_test, predictions))
print("ROC-AUC:", roc_auc_score(y_test, probabilities))
print("\nReporte de clasificación:")
print(classification_report(y_test, predictions, target_names=["No sobrevivió", "Sobrevivió"]))
print("Matriz de confusión:")
print(confusion_matrix(y_test, predictions))

joblib.dump(best_pipeline, "titanic_pipeline.pkl")
print("Pipeline exportado como titanic_pipeline.pkl")
