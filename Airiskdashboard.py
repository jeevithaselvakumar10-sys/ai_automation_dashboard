#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ===================================================================
#   AI AUTOMATION RISK — LINKEDIN-STYLE DASHBOARD (FINAL VERSION)
#   Filename: newtryapp.py
# ===================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score, silhouette_score
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import shap

# ------------------------------
#  PAGE CONFIGURATION
# ------------------------------
st.set_page_config(
    page_title="AI Automation Risk Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
#   GLOBAL CSS – LINKEDIN STYLE
# =====================================================
linkedin_css = """
<style>

html, body, [class*="css"]  {
    font-family: 'Segoe UI', sans-serif;
}

.stApp {
    background-color: #f3f2ef !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e0e0e0;
    padding: 18px 16px;
}

/* Sidebar profile */
.sidebar-profile {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
}

.sidebar-avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background-color: #0A66C2;
    color: #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 18px;
}

.sidebar-name {
    font-weight: 600;
    font-size: 14px;
    color: #111111;
}

.sidebar-role {
    font-size: 12px;
    color: #6b6b6b;
}

.sidebar-divider {
    border: none;
    border-top: 1px solid #e0e0e0;
    margin: 10px 0 14px 0;
}

/* Cards */
.card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
    border: 1px solid #e5e5e5;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.06);
}

/* Badge */
.badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 600;
    color: white;
}

/* Footer */
.footer {
    text-align: center;
    color: #555555;
    margin-top: 30px;
    padding: 15px 0;
    font-size: 14px;
}

</style>
"""
st.markdown(linkedin_css, unsafe_allow_html=True)

# =====================================================
#   JS LOGO HEADER
# =====================================================
js_logo = """
<style>
.logo-container {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 25px;
    margin-top: 5px;
}
.logo-badge {
    width: 46px;
    height: 46px;
    border-radius: 50%;
    background-color: #0A66C2;
    color: white;
    font-weight: 700;
    font-size: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0px 3px 6px rgba(0,0,0,0.2);
}
.logo-title {
    font-size: 24px;
    font-weight: 650;
    color: #1A1A1A;
}
.logo-subtitle {
    font-size: 13px;
    color: #555;
}
</style>

<div class="logo-container">
    <div class="logo-badge">JS</div>
    <div>
        <div class="logo-title">AI Job Automation Risk Insights</div>
        <div class="logo-subtitle">Exploring how AI impacts occupations and skills</div>
    </div>
</div>
"""
st.markdown(js_logo, unsafe_allow_html=True)

# =====================================================
#   SIDEBAR CONTENT (PROFILE, TEAM, UPLOAD, NAV)
# =====================================================
sidebar_profile_html = """
<div class="sidebar-profile">
    <div class="sidebar-avatar">JS</div>
    <div>
        <div class="sidebar-name">Jeevitha Selvakumar</div>
        <div class="sidebar-role">Data Mining Project</div>
    </div>
</div>
<hr class="sidebar-divider">
"""
st.sidebar.markdown(sidebar_profile_html, unsafe_allow_html=True)

st.sidebar.markdown("**Project Team**")
st.sidebar.markdown("• Jeevitha Selvakumar")
st.sidebar.markdown("• Sakshi Chabra")
st.sidebar.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)

st.sidebar.markdown("**Upload Dataset**")
uploaded_data = st.sidebar.file_uploader("automation_processed.csv", type=["csv"])
st.sidebar.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Dataset Explorer",
        "Modeling & Evaluation",
        "Feature Insights",
        "Clustering Analysis",
        "AI Job Automation Risk Insights",
        "Risk Category Overview",
        "Risk Simulator",
        "Job Comparison",
        "Explainability (SHAP Analysis)",
        "Summary & Recommendations"
    ]
)

# =====================================================
#   LOAD DATA & PREP
# =====================================================
if uploaded_data is None:
    st.warning("Please upload automation_processed.csv to continue.")
    st.stop()

df = pd.read_csv(uploaded_data)

# compute enhanced risk
df["Creativity_safe"] = 1 - df["Creativity"]
df["Emotional_safe"] = 1 - df["Emotional_Intelligence"]
df["Analytical_safe"] = 1 - df["Analytical_Thinking"]

df["Enhanced_Automation_Score"] = (
    0.25 * df["Repetitiveness"] +
    0.20 * df["Manual_Dexterity"] +
    0.20 * df["AI_Exposure_Index"] +
    0.15 * df["Skill_Disruption_Index"] +
    0.10 * df["Cognitive_Manual_Ratio"] +
    0.05 * df["Creativity_safe"] +
    0.05 * df["Analytical_safe"]
)

df["Enhanced_Automation_Score"] = (
    df["Enhanced_Automation_Score"] - df["Enhanced_Automation_Score"].min()
) / (df["Enhanced_Automation_Score"].max() - df["Enhanced_Automation_Score"].min())

low_t = df["Enhanced_Automation_Score"].quantile(0.33)
high_t = df["Enhanced_Automation_Score"].quantile(0.66)

def categorize(score):
    if score <= low_t:
        return "Low"
    elif score >= high_t:
        return "High"
    else:
        return "Medium"

df["Automation_Risk"] = df["Enhanced_Automation_Score"].apply(categorize)

encoder = LabelEncoder()
df["Risk_Encoded"] = encoder.fit_transform(df["Automation_Risk"])

features = [
    "Analytical_Thinking", "Emotional_Intelligence", "Creativity",
    "Manual_Dexterity", "AI_Exposure_Index", "Repetitiveness",
    "Skill_Disruption_Index", "Cognitive_Manual_Ratio",
    "Skill_Diversity", "Enhanced_Automation_Score"
]

X = df[features]
y = df["Risk_Encoded"]

# Global Random Forest model for reuse (Feature importance + SHAP)
rf_global = RandomForestClassifier(n_estimators=200, random_state=42)
rf_global.fit(X, y)

# ===================================================================
#   OVERVIEW
# ===================================================================
if menu == "Overview":
    st.markdown("<div class='card'><h3>Project Overview</h3></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Occupations", len(df))
    col2.metric("Risk Categories", df["Automation_Risk"].nunique())
    col3.metric("Clusters (k-means)", 3)

    st.markdown("<div class='card'><h4>Risk Distribution</h4></div>", unsafe_allow_html=True)
    st.bar_chart(df["Automation_Risk"].value_counts())

# ===================================================================
#   DATASET EXPLORER
# ===================================================================
elif menu == "Dataset Explorer":
    st.markdown("<div class='card'><h3>Dataset Explorer</h3></div>", unsafe_allow_html=True)
    st.dataframe(df)

    st.markdown("<div class='card'><h4>Summary Statistics</h4></div>", unsafe_allow_html=True)
    st.dataframe(df.describe())

# ===================================================================
#   MODELING & EVALUATION
# ===================================================================
elif menu == "Modeling & Evaluation":
    st.markdown("<div class='card'><h3>Classification Models</h3></div>", unsafe_allow_html=True)

    lr = LogisticRegression(max_iter=400).fit(X, y)
    rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X, y)

    pred_lr = lr.predict(X)
    pred_rf = rf.predict(X)

    lr_acc = accuracy_score(y, pred_lr)
    rf_acc = accuracy_score(y, pred_rf)

    lr_cm = confusion_matrix(y, pred_lr)
    rf_cm = confusion_matrix(y, pred_rf)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='card'><h4>Logistic Regression</h4></div>", unsafe_allow_html=True)
        st.write(f"Accuracy: **{lr_acc:.3f}**")
        fig, ax = plt.subplots()
        sns.heatmap(lr_cm, annot=True, cmap="Blues", fmt="d",
                    xticklabels=encoder.classes_, yticklabels=encoder.classes_)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with col2:
        st.markdown("<div class='card'><h4>Random Forest</h4></div>", unsafe_allow_html=True)
        st.write(f"Accuracy: **{rf_acc:.3f}**")
        fig, ax = plt.subplots()
        sns.heatmap(rf_cm, annot=True, cmap="Greens", fmt="d",
                    xticklabels=encoder.classes_, yticklabels=encoder.classes_)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

# ===================================================================
#   FEATURE IMPORTANCE
# ===================================================================
elif menu == "Feature Insights":
    st.markdown("<div class='card'><h3>Feature Importance (Random Forest)</h3></div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=rf_global.feature_importances_, y=features, palette="Blues_r")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    st.pyplot(fig)

# ===================================================================
#   CLUSTERING ANALYSIS (PLOTLY)
# ===================================================================
# ===================================================================
#   CLUSTERING ANALYSIS
# ===================================================================
elif menu == "Clustering Analysis":

    st.markdown("<div class='card'><h3>K-Means Clustering</h3></div>", unsafe_allow_html=True)

    # Fit K-means
    kmeans = KMeans(n_clusters=3, random_state=42)
    df["Cluster"] = kmeans.fit_predict(X)

    # Silhouette Score
    sil = silhouette_score(X, df["Cluster"])
    st.metric("Silhouette Score", f"{sil:.3f}")

    # OLD WORKING SCATTERPLOT (matplotlib + seaborn)
    st.markdown("<div class='card'><h4>Cluster Visualization</h4></div>", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        x=df["Repetitiveness"],
        y=df["Cognitive_Manual_Ratio"],
        hue=df["Cluster"],
        palette="deep",
        ax=ax
    )
    ax.set_xlabel("Repetitiveness")
    ax.set_ylabel("Cognitive / Manual Ratio")
    ax.set_title("K-Means Cluster Plot")
    st.pyplot(fig)
# ===================================================================
#   AI JOB AUTOMATION RISK INSIGHTS (JOB-LEVEL VISUALS)
# ===================================================================
elif menu == "AI Job Automation Risk Insights":

    st.markdown("<div class='card'><h3>AI Job Automation Risk Insights</h3></div>", unsafe_allow_html=True)
    st.write("Start typing a job title and select from the dropdown.")

    job_titles = sorted(df["Title"].unique().tolist())
    selected_job = st.selectbox(
        "Search for a Job Title:",
        options=[""] + job_titles,
        index=0
    )

    if selected_job != "":
        job = df[df["Title"] == selected_job].iloc[0]

        # Badge color
        r = job["Automation_Risk"]
        badge_color = "#d11124" if r == "High" else "#f1a100" if r == "Medium" else "#1a7f37"

        # Job card
        st.markdown(
            f"""
            <div class="card">
                <h3>{job['Title']}</h3>
                <p style="color:#666;">O*NET-SOC Code: {job['O*NET-SOC Code']}</p>
                <span class="badge" style="background-color:{badge_color};">Automation Risk: {r}</span>
                <br><br>
                <b>Enhanced Automation Score:</b> {job['Enhanced_Automation_Score']:.3f}
            </div>
            """,
            unsafe_allow_html=True
        )

        # LinkedIn-style risk meter (blue bar)
        st.write("Overall Automation Risk")
        score_val = float(job["Enhanced_Automation_Score"])
        fig, ax = plt.subplots(figsize=(6, 0.5))
        ax.barh([0], [score_val], color="#0A66C2")
        ax.barh([0], [1 - score_val], left=[score_val], color="#E5E5E5")
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["Low", "Medium", "High"])
        for spine in ax.spines.values():
            spine.set_visible(False)
        st.pyplot(fig)

        # --------- VISUAL METRICS USING STREAMLIT PROGRESS BARS ----------
        st.subheader("Simple Progress Bars")

        ai = float(job["AI_Exposure_Index"])
        sd = float(job["Skill_Disruption_Index"])
        rep = float(job["Repetitiveness"])

        metrics_risk = [
            ("AI Exposure Index", ai),
            ("Skill Disruption Index", sd),
            ("Repetitiveness", rep),
        ]

        for label, val in metrics_risk:
            st.write(f"{label}: {val*100:.1f}%")
            st.progress(int(round(val * 100)))

        st.subheader("Human Strength Bars")

        creat = float(job["Creativity"])
        anal = float(job["Analytical_Thinking"])
        emo = float(job["Emotional_Intelligence"])

        metrics_strength = [
            ("Creativity", creat),
            ("Analytical Thinking", anal),
            ("Emotional Intelligence", emo),
        ]

        for label, val in metrics_strength:
            st.write(f"{label}: {val*100:.1f}%")
            st.progress(int(round(val * 100)))

# ===================================================================
#   RISK CATEGORY OVERVIEW (HIGH / MEDIUM / LOW)
# ===================================================================
elif menu == "Risk Category Overview":
    st.markdown("<div class='card'><h3>Risk Category Overview</h3></div>", unsafe_allow_html=True)

    # Quick distribution
    st.markdown("<div class='card'><h4>Risk Distribution</h4></div>", unsafe_allow_html=True)
    st.bar_chart(df["Automation_Risk"].value_counts())

    # Helper to plot top 10 jobs for a given risk level
    def plot_top10(risk_label, color, title):
        subset = df[df["Automation_Risk"] == risk_label].copy()
        if subset.empty:
            st.write(f"No jobs found for {risk_label}.")
            return
        top10 = subset.sort_values("Enhanced_Automation_Score", ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(top10["Title"], top10["Enhanced_Automation_Score"], color=color)
        ax.invert_yaxis()
        ax.set_xlabel("Enhanced Automation Score")
        ax.set_ylabel("Job Title")
        ax.set_title(title)
        st.pyplot(fig)

    # High, Medium, Low cards
    st.markdown("<div class='card'><h4>High-Risk Jobs (Top 10)</h4></div>", unsafe_allow_html=True)
    plot_top10("High", "#d11124", "Top 10 High-Risk Jobs")

    st.markdown("<div class='card'><h4>Medium-Risk Jobs (Top 10)</h4></div>", unsafe_allow_html=True)
    plot_top10("Medium", "#f1a100", "Top 10 Medium-Risk Jobs")

    st.markdown("<div class='card'><h4>Low-Risk Jobs (Top 10)</h4></div>", unsafe_allow_html=True)
    plot_top10("Low", "#1a7f37", "Top 10 Low-Risk Jobs")

# ===================================================================
#   RISK SIMULATOR (WHAT-IF TOOL)
# ===================================================================
elif menu == "Risk Simulator":
    st.markdown("<div class='card'><h3>🔮 Automation Risk Simulator</h3></div>", unsafe_allow_html=True)
    st.write("Adjust the sliders to see how changing skills affects automation risk.")

    col1, col2 = st.columns(2)
    with col1:
        my_rep = st.slider("Repetitiveness", 0.0, 1.0, 0.5)
        my_dex = st.slider("Manual Dexterity", 0.0, 1.0, 0.5)
        my_ai = st.slider("AI Exposure Index", 0.0, 1.0, 0.5)
    with col2:
        my_creat = st.slider("Creativity", 0.0, 1.0, 0.5)
        my_emo = st.slider("Emotional Intelligence", 0.0, 1.0, 0.5)
        my_ana = st.slider("Analytical Thinking", 0.0, 1.0, 0.5)

    safe_creat = 1 - my_creat
    safe_emo = 1 - my_emo
    safe_ana = 1 - my_ana

    # Simplified score (approximation)
    sim_score = (
        0.25 * my_rep +
        0.20 * my_dex +
        0.20 * my_ai +
        0.05 * safe_creat +
        0.05 * safe_ana
    )
    # Normalize roughly between 0 and 1
    sim_score_norm = (sim_score - 0.2) / (0.8 - 0.2)
    sim_score_norm = max(0, min(1, sim_score_norm))

    st.metric("Predicted Automation Risk Score", f"{sim_score_norm:.2f}")

    if sim_score_norm > 0.66:
        st.error("Risk Level: HIGH")
    elif sim_score_norm < 0.33:
        st.success("Risk Level: LOW")
    else:
        st.warning("Risk Level: MEDIUM")

    st.write("Overall Risk (simulated):")
    st.progress(int(round(sim_score_norm * 100)))

# ===================================================================
#   JOB COMPARISON (TWO JOBS SIDE-BY-SIDE)
# ===================================================================
elif menu == "Job Comparison":
    st.markdown("<div class='card'><h3>⚖️ Job Comparison</h3></div>", unsafe_allow_html=True)

    titles = df["Title"].unique().tolist()
    if len(titles) < 2:
        st.write("Need at least 2 jobs in the dataset for comparison.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            job1 = st.selectbox("Select Job 1", titles, index=0)
        with col2:
            job2 = st.selectbox("Select Job 2", titles, index=1)

        data1 = df[df["Title"] == job1].iloc[0]
        data2 = df[df["Title"] == job2].iloc[0]

        colA, colB = st.columns(2)
        with colA:
            st.markdown(f"**{job1}**")
            st.write(f"Risk: {data1['Automation_Risk']}")
            st.write(f"Score: {data1['Enhanced_Automation_Score']:.3f}")
            st.progress(int(round(data1["Enhanced_Automation_Score"] * 100)))
        with colB:
            st.markdown(f"**{job2}**")
            st.write(f"Risk: {data2['Automation_Risk']}")
            st.write(f"Score: {data2['Enhanced_Automation_Score']:.3f}")
            st.progress(int(round(data2["Enhanced_Automation_Score"] * 100)))

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Skill Profile Comparison (Radar)")

        categories = ['Creativity', 'Emotional_Intelligence',
                      'Analytical_Thinking', 'Manual_Dexterity', 'Repetitiveness']

        r1 = [data1[c] for c in categories]
        r2 = [data2[c] for c in categories]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=r1,
            theta=categories,
            fill='toself',
            name=job1,
            line_color='#0A66C2'
        ))
        fig.add_trace(go.Scatterpolar(
            r=r2,
            theta=categories,
            fill='toself',
            name=job2,
            line_color='#E53935'
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

# ===================================================================
#   EXPLAINABILITY (SHAP ANALYSIS) – FIXED
# ===================================================================
elif menu == "Explainability (SHAP Analysis)":
    st.markdown("<div class='card'><h3>🧠 Explainability (SHAP Analysis)</h3></div>", unsafe_allow_html=True)

    st.write("Global feature impact for the **Random Forest** model, focusing on the **High-risk** class.")

    # Create TreeExplainer
    explainer = shap.TreeExplainer(rf_global)
    shap_values = explainer.shap_values(X)

    # ---- HANDLE MULTI-CLASS SHAP RETURNS SAFELY ----
    if isinstance(shap_values, list):
        # pick High class if available, else first class
        if "High" in encoder.classes_:
            high_idx = list(encoder.classes_).index("High")
        else:
            high_idx = 0
        sv = shap_values[high_idx]          # shape = (n_samples, n_features)
    else:
        sv = shap_values                    # binary classification case

    # sv MUST be 2D: (samples, features)
    if sv.ndim == 3:
        # collapse unexpected extra dimension
        sv = sv.mean(axis=2)

    # Now compute mean |SHAP|
    mean_abs_shap = np.mean(np.abs(sv), axis=0)   # 1D array

    # Build Series safely
    shap_importance = pd.Series(mean_abs_shap, index=features).sort_values()

    # ---- PLOT ----
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(shap_importance.index, shap_importance.values, color="#0A66C2")
    ax.set_xlabel("Mean |SHAP value| (impact on risk prediction)")
    ax.set_ylabel("Feature")
    ax.set_title("Global Feature Impact for High Automation Risk")
    st.pyplot(fig)

# ===================================================================
#   SUMMARY & RECOMMENDATIONS (WITH CARDS)
# ===================================================================
elif menu == "Summary & Recommendations":
    st.markdown("<div class='card'><h3>Summary & Recommendations</h3></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='card'><h4>Automation Risk Distribution</h4></div>", unsafe_allow_html=True)
        st.bar_chart(df["Automation_Risk"].value_counts())

    with col2:
        if "Cluster" in df.columns:
            st.markdown("<div class='card'><h4>Clusters by Risk Category</h4></div>", unsafe_allow_html=True)
            cluster_pivot = pd.crosstab(df["Cluster"], df["Automation_Risk"])
            st.dataframe(cluster_pivot)

    st.markdown("### Recommendations")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class='card'>
            <h4>Reduce Automation Risk</h4>
            <ul>
                <li>Shift workers from repetitive tasks to higher-cognitive duties.</li>
                <li>Use automation for routine work, but keep humans in decision roles.</li>
                <li>Strengthen roles with high creativity and analytical thinking.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class='card'>
            <h4>Workforce Skills Strategy</h4>
            <ul>
                <li>Prioritize reskilling in AI literacy and data-driven thinking.</li>
                <li>Invest in emotional intelligence and collaboration training.</li>
                <li>Use risk scores to design training paths for at-risk roles.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            """
            <div class='card'>
            <h4>Organizational Planning</h4>
            <ul>
                <li>Use high-risk clusters to identify where job redesign is needed.</li>
                <li>Leverage low-risk roles to supervise and audit automated systems.</li>
                <li>Monitor risk categories over time as AI tools are introduced.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

# ===================================================================
#   FOOTER
# ===================================================================
st.markdown(
    """<div class='footer'>
    © 2025 — Created by <b>Jeevitha Selvakumar</b> & <b>Sakshi Chabra</b>
    </div>""",
    unsafe_allow_html=True
)

