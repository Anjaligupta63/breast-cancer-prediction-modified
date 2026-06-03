import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

# Path to dataset
FILE_PATH = r"C:\Users\anjal\OneDrive\Desktop\Breast Cancer Prediction\breast-cancer-data.csv"


def main():
    st.title("Breast Cancer Recurrence Prediction")

    # Load dataset
    data = pd.read_csv(FILE_PATH)

    st.write("Dataset Preview:")
    st.write(data.head())

    # Remove unwanted quotes from string columns
    for column in data.select_dtypes(include=["object"]).columns:
        data[column] = data[column].str.strip("'")

    # Features and target
    X = data.drop("class", axis=1)
    y = data["class"].map(
        lambda x: 1 if x == "recurrence-events" else 0
    )

    # One-hot encoding
    categorical_features = X.columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first"), categorical_features)
        ]
    )

    # Pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("scaler", StandardScaler(with_mean=False)),
            ("classifier", LogisticRegression(max_iter=1000))
        ]
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # Train model
    pipeline.fit(X_train, y_train)

    # Accuracy
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.write(f"Model Accuracy: {accuracy:.2f}")

    st.subheader("Enter Patient Details")

    age = st.selectbox("Age", sorted(X["age"].unique()))
    menopause = st.selectbox("Menopause", sorted(X["menopause"].unique()))
    tumor_size = st.selectbox("Tumor Size", sorted(X["tumor-size"].unique()))
    inv_nodes = st.selectbox("Involved Nodes", sorted(X["inv-nodes"].unique()))
    node_caps = st.selectbox("Node Caps", sorted(X["node-caps"].unique()))
    deg_malig = st.selectbox("Degree of Malignancy", sorted(X["deg-malig"].unique()))
    breast = st.selectbox("Breast", sorted(X["breast"].unique()))
    breast_quad = st.selectbox("Breast Quadrant", sorted(X["breast-quad"].unique()))
    irradiate = st.selectbox("Irradiate", sorted(X["irradiate"].unique()))

    if st.button("Predict"):

        input_data = pd.DataFrame({
            "age": [age],
            "menopause": [menopause],
            "tumor-size": [tumor_size],
            "inv-nodes": [inv_nodes],
            "node-caps": [node_caps],
            "deg-malig": [deg_malig],
            "breast": [breast],
            "breast-quad": [breast_quad],
            "irradiate": [irradiate]
        })

        prediction = pipeline.predict(input_data)

        if prediction[0] == 1:
            st.error("Predicted: Recurrence-Events (Cancerous)")
        else:
            st.success("Predicted: No-Recurrence-Events (Non-Cancerous)")


if __name__ == "__main__":
    main()