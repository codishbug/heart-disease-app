import streamlit as st
import pickle
import numpy as np
import datetime
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# -----------------------------
# Load the saved model
# -----------------------------
model = pickle.load(open("model.pkl", "rb"))

st.title("🫀 Heart Disease Prediction")

st.markdown("""
Fill in the details below. The app will predict whether you may be at risk of heart disease.  
*(This is not a medical diagnosis. Please consult a doctor for accurate advice.)*
""")

# -----------------------------
# Collect input features
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", 20, 100, value=None, placeholder="Enter your age")
    sex = st.selectbox("Biological Sex", ["Select", "Female", "Male"])
    sex = None if sex == "Select" else (0 if sex == "Female" else 1)
    cp = st.selectbox("Chest Pain Experience", 
                      ["Select",
                       "Typical Angina (chest pain on exertion)", 
                       "Atypical Angina (chest pain not related to exertion)",
                       "Non-heart-related chest pain", 
                       "No chest pain"])
    cp = None if cp == "Select" else [
          "Typical Angina (chest pain on exertion)", 
          "Atypical Angina (chest pain not related to exertion)", 
          "Non-heart-related chest pain", 
          "No chest pain"].index(cp)
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, value=120)
    chol = st.number_input("Cholesterol Level (mg/dl)", 100, 600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar", ["≤120 mg/dl (Normal)", ">120 mg/dl (High)"], index=0)
    fbs = 0 if fbs == "≤120 mg/dl (Normal)" else 1
    family_history = st.selectbox("Family history of heart disease?", ["No", "Yes"])
    family_history = 1 if family_history == "Yes" else 0

with col2:
    restecg = st.selectbox("ECG Result", 
                           ["Normal", 
                            "Minor irregularities", 
                            "Possible heart muscle enlargement"], index=0)
    restecg = ["Normal", "Minor irregularities", "Possible heart muscle enlargement"].index(restecg)
    thalach = st.number_input("Maximum Heart Rate Achieved", 70, 220, value=150)
    exang = st.selectbox("Chest Pain During Exercise", ["Select", "No", "Yes"])
    exang = None if exang == "Select" else (0 if exang == "No" else 1)
    oldpeak = st.number_input("ST Depression (ECG reading)", 0.0, 10.0, value=1.0, step=0.1)
    slope = st.selectbox("Heart Rate Response Slope During Exercise", ["Select", "Gradually increases", "No change", "Decreases"])
    slope = None if slope == "Select" else ["Gradually increases", "No change", "Decreases"].index(slope)
    ca = st.selectbox("Number of Major Heart Arteries with Narrowing (0–3)", [0, 1, 2, 3], index=0)
    thal = st.selectbox("Blood Flow Test Result", ["Select", "Normal", "Permanent problem", "Temporary problem"])
    thal = None if thal == "Select" else ["Normal", "Permanent problem", "Temporary problem"].index(thal) + 1

# -----------------------------
# Lifestyle Factors
# -----------------------------
st.subheader("💡 Lifestyle Factors")
smoking = st.checkbox("Do you smoke?")
alcohol = st.checkbox("Do you consume alcohol?")
physical_activity = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])
stress = st.selectbox("Stress Level", ["Low", "Moderate", "High"])

# -----------------------------
# Prediction + Report Generation
# -----------------------------
if st.button("📄 Generate Report"):
    if None in [age, sex, cp, exang, slope, thal]:
        st.error("⚠️ Please fill in all fields before generating the report.")
    else:
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                                restecg, thalach, exang, oldpeak,
                                slope, ca, thal]])
        prediction = model.predict(input_data)[0]

        # Result message
        if prediction == 1:
            st.success("✅ Low Risk: You are unlikely to have heart disease.")
        else:
            st.error("⚠️ High Risk: You may have heart disease. Consult a doctor.")

        # -----------------------------
        # Detailed Analysis
        # -----------------------------
        st.subheader("📊 Detailed Analysis")
        risk_factors = []
        weights = []
        age_note = ""
        if age > 50: age_note = "Age can increase risk; regular check-ups are recommended."
        if trestbps > 130: risk_factors.append("High blood pressure"); weights.append(3)
        if chol > 240: risk_factors.append("High cholesterol"); weights.append(3)
        if fbs == 1: risk_factors.append("High blood sugar"); weights.append(2)
        if exang == 1: risk_factors.append("Chest pain during exercise"); weights.append(2)
        if oldpeak > 2: risk_factors.append("Significant ECG changes (possible heart stress)"); weights.append(4)
        if family_history == 1: risk_factors.append("Family history of heart disease"); weights.append(2)
        if smoking: risk_factors.append("Smoking increases risk"); weights.append(2)
        if alcohol: risk_factors.append("Alcohol consumption increases risk"); weights.append(1)
        if physical_activity == "Low": risk_factors.append("Low physical activity"); weights.append(2)
        if stress == "High": risk_factors.append("High stress levels"); weights.append(1)

        if len(risk_factors) == 0 and not age_note: st.info("No major risk factors detected from inputs.")
        else:
            for rf in risk_factors: st.write("- ", rf)
            if age_note: st.info(age_note)

        # -----------------------------
        # Visualization Charts
        # -----------------------------
        st.subheader("📈 Visualization of Key Health Indicators")
        fig, ax = plt.subplots()
        features = ["Age", "Resting BP", "Cholesterol", "Max HR", "ECG ST Depression"]
        values = [age, trestbps, chol, thalach, oldpeak]
        ax.bar(features, values, color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336'])
        ax.set_ylabel("Value")
        ax.set_title("Patient Health Indicators")
        st.pyplot(fig)

        if prediction == 0 and risk_factors:
            fig2, ax2 = plt.subplots()
            if len(risk_factors) == 1:
                ax2.pie([weights[0], sum(weights)], labels=[risk_factors[0], "Other"], autopct='%1.1f%%', startangle=90, colors=["#FF7043", "#E0E0E0"])
            else:
                ax2.pie(weights, labels=risk_factors, autopct='%1.1f%%', startangle=90)
            ax2.set_title("Risk Factor Severity Contribution")
            st.pyplot(fig2)

        # -----------------------------
        # Recommendations
        # -----------------------------
        st.subheader("🩺 Recommendations")
        st.markdown("""
        - Maintain a balanced diet low in saturated fats and cholesterol.  
        - Engage in regular physical activity (consult your doctor first).  
        - Monitor blood pressure, blood sugar, and cholesterol regularly.  
        - Avoid smoking and limit alcohol intake.  
        - Manage stress with relaxation techniques.  
        - Schedule regular health check-ups.  
        """)

        # -----------------------------
        # Useful Resources
        # -----------------------------
        st.subheader("📌 Useful Resources")
        st.markdown("""
        - [World Health Organization (WHO)](https://www.who.int/india/health-topics/cardiovascular-diseases)  
        - [World Heart Federation](https://world-heart-federation.org/)  
        - [Healthy Lifestyle Guide](https://www.cdc.gov/heart-disease/prevention.htm)  
        """)

        # -----------------------------
        # PDF Report Generation
        # -----------------------------
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        elements = []

        # Custom styles
        title_style = ParagraphStyle('TitleStyle', parent=styles['Title'], fontSize=24, textColor=colors.HexColor('#1A5276'), alignment=1)
        risk_heading_style = ParagraphStyle('RiskHeadingStyle', parent=styles['Heading2'], fontSize=16, textColor=colors.HexColor('#D35400'))
        rec_heading_style = ParagraphStyle('RecHeadingStyle', parent=styles['Heading2'], fontSize=16, textColor=colors.HexColor('#5B2C6F'))
        resource_heading_style = ParagraphStyle('ResourceHeadingStyle', parent=styles['Heading2'], fontSize=16, textColor=colors.HexColor('#1F618D'))
        normal_style = ParagraphStyle('NormalStyle', parent=styles['Normal'], fontSize=12, textColor=colors.black)

        # Title
        elements.append(Paragraph("Heart Disease Prediction Report", title_style))
        elements.append(Spacer(1, 12))
        elements.append(Spacer(1, 12))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"Date: {datetime.date.today()}", normal_style))
        elements.append(Spacer(1, 12))
        
        # Prediction
        prediction_color = colors.green if prediction == 1 else colors.red
        prediction_style = ParagraphStyle('PredictionStyle', parent=styles['Heading2'], fontSize=16, textColor=prediction_color, alignment=1)
        elements.append(Paragraph(f"Prediction: {'Low Risk' if prediction==1 else 'High Risk'}", prediction_style))
        elements.append(Spacer(1, 12))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#BDC3C7")))
        elements.append(Spacer(1, 12))

        # Patient Details Table
        data = [
            ["Attribute", "Value"],
            ["Age", age],
            ["Sex", "Male" if sex==1 else "Female"],
            ["Resting BP (mmHg)", trestbps],
            ["Cholesterol (mg/dl)", chol],
            ["Max Heart Rate", thalach],
            ["Fasting Blood Sugar", ">120 mg/dl" if fbs==1 else "≤120 mg/dl"],
            ["Family History", "Yes" if family_history==1 else "No"],
            ["Smoking", "Yes" if smoking else "No"],
            ["Alcohol", "Yes" if alcohol else "No"],
            ["Physical Activity", physical_activity],
            ["Stress Level", stress],
        ]
        table = Table(data, hAlign='LEFT')
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#4CAF50")),
            ('TEXTCOLOR',(0,0),(-1,0),colors.white),
            ('ALIGN',(0,0),(-1,-1),'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 12),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#D5F5E3")),
            ('GRID', (0,0), (-1,-1), 0.5, colors.black),
            ('BOTTOMPADDING', (0,0), (-1,0), 8)
        ]))
        elements.append(table)
        elements.append(Spacer(1, 12))

        # Risk Factors
        elements.append(Paragraph("Risk Factors:", risk_heading_style))
        if risk_factors:
            for rf in risk_factors:
                elements.append(Paragraph(f"- {rf}", normal_style))
        else:
            elements.append(Paragraph("None detected", normal_style))
        if age_note:
            elements.append(Paragraph(f"- {age_note}", normal_style))
        elements.append(Spacer(1, 12))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#BDC3C7")))
        elements.append(Spacer(1, 12))

        # Recommendations
        elements.append(Paragraph("Recommendations:", rec_heading_style))
        recs = [
            "Maintain a balanced diet low in saturated fats and cholesterol.",
            "Engage in regular physical activity.",
            "Monitor blood pressure, blood sugar, and cholesterol regularly.",
            "Avoid smoking and limit alcohol intake.",
            "Manage stress with relaxation techniques.",
            "Schedule regular health check-ups."
        ]
        for r in recs: elements.append(Paragraph(f"- {r}", normal_style))
        elements.append(Spacer(1, 12))

        # Charts
        img_buffer = BytesIO()
        fig.savefig(img_buffer, format='PNG', bbox_inches='tight')
        img_buffer.seek(0)
        fig_width, fig_height = fig.get_size_inches()
        aspect_ratio = fig_height / fig_width
        img_width = 400
        img_height = img_width * aspect_ratio
        elements.append(Image(img_buffer, width=img_width, height=img_height))

        if prediction == 0 and risk_factors:
            img_buffer2 = BytesIO()
            fig2.savefig(img_buffer2, format='PNG', bbox_inches='tight')
            img_buffer2.seek(0)
            elements.append(Image(img_buffer2, width=400, height=200))
            elements.append(Spacer(1, 12))
        
        # Resources
        elements.append(Paragraph("Useful Resources:", resource_heading_style))
        
        # Style for normal text
        normal_text_style = ParagraphStyle(
            'NormalTextStyle',
            parent=styles['Normal'],
            fontSize=12,
            textColor=colors.black
        )
        
        # Resources with clickable-looking URLs
        resources = [
            ("World Health Organization (WHO)", "https://www.who.int/india/health-topics/cardiovascular-diseases"),
            ("World Heart Federation", "https://world-heart-federation.org/"),
            ("Healthy Lifestyle Guide", "https://www.cdc.gov/heart-disease/prevention/index.html")
        ]
        
        for name, url in resources:
            # Use mini-HTML for URL styling
            resource_line = f"{name}: <u><font color='#1F618D'>{url}</font></u>"
            elements.append(Paragraph(resource_line, normal_text_style))
            elements.append(Spacer(1, 6))


        doc.build(elements)
        buffer.seek(0)

        st.download_button("⬇️ Download PDF Report", buffer, file_name="heart_report.pdf", mime="application/pdf")
