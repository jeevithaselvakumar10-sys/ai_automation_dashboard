# AI Automation Risk Analysis Dashboard

An interactive Streamlit-based analytics dashboard designed to explore how Artificial Intelligence impacts job automation risk across hundreds of occupations.  
This project integrates **Machine Learning**, **Explainable AI (SHAP)**, **Interactive Visualizations**, and **User-Friendly Insights** to help users understand where automation may disrupt the workforce.

---

## 🚀 Project Features

### 🔍 Job Automation Risk Explorer  
- Search any job title  
- View complete automation-risk profile  
- Visual bar indicators for:  
  - AI Exposure Index  
  - Skill Disruption Index  
  - Repetitiveness  
  - Creativity  
  - Emotional Intelligence  
  - Analytical Thinking  
- Enhanced Automation Score (0–1 scale)  
- Risk categories: **Low**, **Medium**, **High**

---

### 📊 Risk Category Overview  
- Top 10 **High-risk**, **Medium-risk**, and **Low-risk** jobs  
- Horizontal bar charts for visual comparison  

---

### 🤖 Machine Learning Modeling  
Includes:  
- Logistic Regression  
- Random Forest Classifier  
- Confusion Matrices  
- Accuracy Comparison  

---

### 🌌 Explainability (SHAP)  
- Global feature importance  
- Visual explanation of how each feature impacts automation risk  

---

### 🧩 Clustering Analysis  
- K-Means clustering (k=3)  
- Visualization of job groups using:  
  - Repetitiveness  
  - Cognitive/Manual Ratio  

---

### 🧪 Dataset Explorer  
- Full dataset preview  
- Descriptive statistics  
- Interactive filtering  

---

## 🧠 Tech Stack

| Component | Tools Used |
|----------|------------|
| Programming Language | Python |
| Dashboard Framework | Streamlit |
| ML / Statistics | scikit-learn, pandas, numpy |
| Visualization | Plotly, Matplotlib, Seaborn |
| Explainability | SHAP |
| Deployment Ready | Yes ✔️ |

---

## 📂 Project Structure

```
📁 ai_automation_dashboard/
│── lethisbefinal.py         # Main Streamlit App
│── automation_processed.csv # Dataset used
│── README.md                # Documentation
│── requirements.txt         # Libraries list
│── /images                  # Screenshots (optional)
```

---

## ▶️ How to Run the Project

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Run the Streamlit app
```bash
streamlit run lethisbefinal.py
```

### 3️⃣ View the dashboard  
It will open automatically at:
```
http://localhost:8501
```

---

## 🌟 Contributors

| Name | Role |
|------|------|
| **Jeevitha Selvakumar** | 
| **Sakshi Chabra** | 

---

## ⭐ Support the Project
If you found this project helpful, please **star the repository on GitHub!** 🌟

