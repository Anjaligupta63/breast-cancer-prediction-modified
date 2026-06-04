import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ----------------------------
# PAGE CONFIG
# ----------------------------

st.set_page_config(
    page_title="Breast Cancer Hospital Dashboard",
    page_icon="🎀",
    layout="wide"
)

# ----------------------------
# THEME TOGGLE
# ----------------------------

theme = st.sidebar.radio(
    "Theme",
    ["Light Mode", "Dark Mode"]
)

if theme == "Light Mode":
    bg_color = "#ffe6e6"
    card_color = "#ffffff"
    text_color = "#000000"
else:
    bg_color = "#1e1e1e"
    card_color = "#2d2d2d"
    text_color = "#ffffff"

# ----------------------------
# CUSTOM CSS
# ----------------------------

st.markdown(f"""
<style>

.stApp {{
    background-color:{bg_color};
}}

.main-header {{
    background:#cc0000;
    padding:20px;
    border-radius:15px;
    text-align:center;
}}

.main-header h1 {{
    color:gold;
}}

.card {{
    background:{card_color};
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.2);
    transition:0.3s;
}}

.card:hover {{
    transform:scale(1.02);
}}

.footer {{
    background:#cc0000;
    padding:15px;
    border-radius:15px;
    text-align:center;
}}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# HEADER
# ----------------------------

st.markdown("""
<div class="main-header">
<h1>🎀 Breast Cancer Hospital Dashboard</h1>
<h4>Machine Learning Based Prediction System</h4>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# SIDEBAR LOGO
# ----------------------------

st.sidebar.markdown("# 🎀 Hospital Panel")

# ----------------------------
# LOAD DATA
# ----------------------------

FILE_PATH = "breast-cancer-data.csv"
data = pd.read_csv(FILE_PATH, sep="\t")
data.columns = (
    data.columns
    .str.strip()
    .str.lower()
)

for col in data.columns:
    data[col] = (
        data[col]
        .astype(str)
        .str.strip("'")
    )

# ----------------------------
# PREPARE DATA
# ----------------------------

X = data.drop(
    "class",
    axis=1
)

y = data["class"].apply(
    lambda x:
    1 if x == "recurrence-events"
    else 0
)
# ----------------------------
# TRAIN MODEL
# ----------------------------

categorical_features = X.columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

y_pred = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

# ----------------------------
# DASHBOARD STATISTICS
# ----------------------------

total_patients = len(data)

recurrence_cases = int(
    y.sum()
)

healthy_cases = (
    total_patients -
    recurrence_cases
)

# ----------------------------
# ANIMATED CARDS
# ----------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown(f"""
    <div class="card">
    <h3>👩 Patients</h3>
    <h1>{total_patients}</h1>
    </div>
    """,
    unsafe_allow_html=True)

with col2:

    st.markdown(f"""
    <div class="card">
    <h3>⚠ Recurrence</h3>
    <h1>{recurrence_cases}</h1>
    </div>
    """,
    unsafe_allow_html=True)

with col3:

    st.markdown(f"""
    <div class="card">
    <h3>✅ Healthy</h3>
    <h1>{healthy_cases}</h1>
    </div>
    """,
    unsafe_allow_html=True)

with col4:

    st.markdown(f"""
    <div class="card">
    <h3>📊 Accuracy</h3>
    <h1>{accuracy*100:.2f}%</h1>
    </div>
    """,
    unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------
# PLOTLY GAUGE METER
# ----------------------------

st.subheader(
    "🎯 Model Accuracy Gauge"
)

gauge = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=accuracy * 100,
        title={
            "text":
            "Model Accuracy (%)"
        },
        gauge={
            "axis": {
                "range": [0, 100]
            },
            "bar": {
                "color": "red"
            },
            "steps": [
                {
                    "range": [0, 50],
                    "color": "#ffcccc"
                },
                {
                    "range": [50, 75],
                    "color": "#ffe6b3"
                },
                {
                    "range": [75, 100],
                    "color": "#ccffcc"
                }
            ]
        }
    )
)

st.plotly_chart(
    gauge,
    use_container_width=True
)

# ----------------------------
# DATA DISTRIBUTION PIE CHART
# ----------------------------

st.subheader(
    "📈 Dataset Distribution"
)

pie_data = pd.DataFrame(
    {
        "Category":
        [
            "Recurrence",
            "Healthy"
        ],

        "Count":
        [
            recurrence_cases,
            healthy_cases
        ]
    }
)

fig = px.pie(
    pie_data,
    values="Count",
    names="Category",
    title="Patient Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)
# ----------------------------
# SINGLE PATIENT PREDICTION
# ----------------------------

st.markdown("---")

st.header(
    "🧑 Patient Prediction Center"
)

st.sidebar.header(
    "📝 Enter Patient Details"
)

age = st.sidebar.selectbox(
    "Age",
    sorted(X["age"].unique())
)

menopause = st.sidebar.selectbox(
    "Menopause",
    sorted(X["menopause"].unique())
)

tumor_size = st.sidebar.selectbox(
    "Tumor Size",
    sorted(X["tumer-size"].unique())
)

inv_nodes = st.sidebar.selectbox(
    "Involved Nodes",
    sorted(X["inv-nodes"].unique())
)

node_caps = st.sidebar.selectbox(
    "Node Caps",
    sorted(X["node-caps"].replace("?", None).dropna().astype(str).unique())
)

deg_malig = st.sidebar.selectbox(
    "Degree of Malignancy",
    sorted(X["deg-malig"].unique())
)

breast = st.sidebar.selectbox(
    "Breast",
    sorted(X["breast"].unique())
)

breast_quad = st.sidebar.selectbox(
    "Breast Quadrant",
   sorted(X["breast-quad"].dropna().astype(str).unique())
)

irradiate = st.sidebar.selectbox(
    "Irradiate",
    sorted(X["irradiate"].unique())
)

predict_btn = st.sidebar.button(
    "🔍 Predict Cancer Risk"
)

if predict_btn:

    patient = pd.DataFrame(
        {
            "age":[age],
            "menopause":[menopause],
            "tumer-size":[tumor_size],
            "inv-nodes":[inv_nodes],
            "node-caps":[node_caps],
            "deg-malig":[deg_malig],
            "breast":[breast],
            "breast-quad":[breast_quad],
            "irradiate":[irradiate]
        }
    )

    prediction = model.predict(
        patient
    )[0]

    probability = model.predict_proba(
        patient
    )[0]

    recurrence_prob = (
        probability[1] * 100
    )

    healthy_prob = (
        probability[0] * 100
    )

    st.subheader(
        "Prediction Result"
    )

    # ----------------------------
    # RISK GAUGE
    # ----------------------------

    risk_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=recurrence_prob,
            title={
                "text":"Cancer Recurrence Risk (%)"
            },
            gauge={
                "axis":{
                    "range":[0,100]
                },
                "bar":{
                    "color":"darkred"
                },
                "steps":[
                    {
                        "range":[0,30],
                        "color":"lightgreen"
                    },
                    {
                        "range":[30,70],
                        "color":"khaki"
                    },
                    {
                        "range":[70,100],
                        "color":"lightcoral"
                    }
                ]
            }
        )
    )

    st.plotly_chart(
        risk_gauge,
        use_container_width=True
    )

    # ----------------------------
    # CONFIDENCE CARD
    # ----------------------------

    confidence = max(
        healthy_prob,
        recurrence_prob
    )

    st.metric(
        "Confidence Score",
        f"{confidence:.2f}%"
    )

    # ----------------------------
    # RESULT DISPLAY
    # ----------------------------

    if prediction == 1:

        st.markdown(
        f"""
        <div style="
        background:#ffcccc;
        padding:20px;
        border-radius:15px;
        text-align:center;">
        <h2>
        ⚠ High Risk of Cancer Recurrence
        </h2>
        <h3>
        Risk Score:
        {recurrence_prob:.2f}%
        </h3>
        </div>
        """,
        unsafe_allow_html=True
        )

        st.warning(
            """
            Patient may have a higher chance
            of recurrence.

            Recommendation:
            Consult an oncologist and
            schedule further examinations.
            """
        )

    else:

        st.markdown(
        f"""
        <div style="
        background:#d4edda;
        padding:20px;
        border-radius:15px;
        text-align:center;">
        <h2>
        ✅ Low Risk of Cancer Recurrence
        </h2>
        <h3>
        Safety Score:
        {healthy_prob:.2f}%
        </h3>
        </div>
        """,
        unsafe_allow_html=True
        )

        st.success(
            """
            Patient currently appears
            at lower recurrence risk.

            Continue regular health checkups.
            """
        )

    # ----------------------------
    # PATIENT SUMMARY TABLE
    # ----------------------------

    st.subheader(
        "📋 Patient Summary"
    )

    summary = patient.copy()

    summary["Prediction"] = (
        "Recurrence"
        if prediction == 1
        else "No Recurrence"
    )

    summary["Confidence"] = (
        f"{confidence:.2f}%"
    )

    st.dataframe(
        summary,
        use_container_width=True
    )
    # ----------------------------
# BULK CSV PREDICTION
# ----------------------------

st.markdown("---")

st.header(
    "📁 Bulk Patient Prediction"
)

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        bulk_data = pd.read_csv(uploaded_file, sep="\t")

        st.subheader(
            "Uploaded Data Preview"
        )

        st.dataframe(
            bulk_data.head(),
            use_container_width=True
        )

        required_columns = [
            "age",
            "menopause",
            "tumer-size",
            "inv-nodes",
            "node-caps",
            "deg-malig",
            "breast",
            "breast-quad",
            "irradiate"
        ]

        missing_cols = [
            col
            for col in required_columns
            if col not in bulk_data.columns
        ]

        if len(missing_cols) > 0:

            st.error(
                f"Missing Columns: {missing_cols}"
            )

        else:

            predictions = model.predict(
                bulk_data
            )

            probabilities = (
                model.predict_proba(
                    bulk_data
                )
            )

            bulk_data["Prediction"] = [
                "Recurrence"
                if p == 1
                else "No Recurrence"
                for p in predictions
            ]

            bulk_data["Confidence"] = [
                round(
                    max(prob) * 100,
                    2
                )
                for prob in probabilities
            ]

            st.success(
                "Prediction Completed Successfully"
            )

            st.subheader(
                "Prediction Results"
            )

            st.dataframe(
                bulk_data,
                use_container_width=True
            )

            # ----------------------------
            # DOWNLOAD BUTTON
            # ----------------------------

            csv_file = (
                bulk_data
                .to_csv(
                    index=False
                )
                .encode("utf-8")
            )

            st.download_button(
                label="⬇ Download Results CSV",
                data=csv_file,
                file_name="prediction_results.csv",
                mime="text/csv"
            )

            # ----------------------------
            # BULK ANALYTICS
            # ----------------------------

            st.subheader(
                "📊 Bulk Prediction Analytics"
            )

            recurrence_count = (
                bulk_data["Prediction"]
                .value_counts()
                .get(
                    "Recurrence",
                    0
                )
            )

            healthy_count = (
                bulk_data["Prediction"]
                .value_counts()
                .get(
                    "No Recurrence",
                    0
                )
            )

            analytics_df = pd.DataFrame(
                {
                    "Category": [
                        "Recurrence",
                        "No Recurrence"
                    ],
                    "Count": [
                        recurrence_count,
                        healthy_count
                    ]
                }
            )

            # ----------------------------
            # PIE CHART
            # ----------------------------

            pie_chart = px.pie(
                analytics_df,
                names="Category",
                values="Count",
                title="Prediction Distribution"
            )

            st.plotly_chart(
                pie_chart,
                use_container_width=True
            )

            # ----------------------------
            # BAR CHART
            # ----------------------------

            bar_chart = px.bar(
                analytics_df,
                x="Category",
                y="Count",
                title="Prediction Counts"
            )

            st.plotly_chart(
                bar_chart,
                use_container_width=True
            )

            # ----------------------------
            # HOSPITAL STATISTICS PANEL
            # ----------------------------

            st.subheader(
                "🏥 Hospital Statistics"
            )

            h1, h2, h3 = st.columns(3)

            with h1:

                st.metric(
                    "Total Uploaded Patients",
                    len(
                        bulk_data
                    )
                )

            with h2:

                st.metric(
                    "High Risk Patients",
                    recurrence_count
                )

            with h3:

                st.metric(
                    "Low Risk Patients",
                    healthy_count
                )

    except Exception as e:

        st.error(
            f"Error: {e}"
        )

# ----------------------------
# DATASET PREVIEW
# ----------------------------

st.markdown("---")

st.header(
    "📋 Dataset Preview"
)

st.dataframe(
    data.head(),
    use_container_width=True
)

# ----------------------------
# DATASET INFORMATION
# ----------------------------

st.subheader(
    "Dataset Information"
)

info1, info2, info3 = st.columns(3)

with info1:

    st.metric(
        "Rows",
        data.shape[0]
    )

with info2:

    st.metric(
        "Columns",
        data.shape[1]
    )

with info3:

    st.metric(
        "Accuracy",
        f"{accuracy*100:.2f}%"
    )

# ----------------------------
# FOOTER
# ----------------------------

st.markdown("---")

st.markdown(
"""
<div style="
background:#cc0000;
padding:20px;
border-radius:15px;
text-align:center;
">

<h2 style="color:gold;">
Developed By
</h2>

<h3 style="color:white;">
Anjali Gupta
</h3>

<h4 style="color:white;">
Indira Gandhi Delhi Technical University
for Women (IGDTUW)
</h4>

<p style="color:white;">
Breast Cancer Recurrence Prediction
Using Machine Learning
</p>

</div>
""",
unsafe_allow_html=True
)