# Screenshot Guide

Take these 4 screenshots and save them here with the exact filenames below.
All images should be **1400px wide minimum**, saved as PNG.

---

## 01_risk_assessment.png
**What:** Streamlit app → "Customer Risk Assessment" tab
- Fill in: tenure=6, monthly=85, contract=Month-to-Month, internet=Fiber, no tech support, no online security
- Click "Predict Churn Risk"
- Screenshot should show the gauge chart at ~70%+ churn probability
- The recommendation box ("URGENT: Contact within 30 days") should be visible

## 02_cluster_scatter.png
**What:** Streamlit app → "Segment Overview" tab
- Default view — the Tenure vs Monthly Charges scatter plot
- All 4 clusters should be visible (green, blue, amber, red dots)
- The segment summary table should be visible on the right

## 03_revenue_impact.png
**What:** Streamlit app → "Revenue Impact" tab
- Set: top 20% at-risk, 35% retention rate, €1440 avg annual revenue
- The 4 KPI tiles should show revenue numbers
- The area chart below should be visible

## 04_model_performance.png
**What:** Streamlit app → "Model Performance" tab
- Default view — the histogram showing churn probability distribution
- Green (retained) and red (churned) bars should be clearly separated
- The decision threshold line at 0.5 should be visible

---

After adding all screenshots, commit and push. The README already references these filenames.
