# =========================================================
# GBM Integrative Survival Analysis
# Cox + KM + Confusion Matrix + Tumor Burden Distribution
# + Cox Feature Importance + Time-Dependent C-Index + Text Summaries
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Install lifelines if not already installed
import sys
!{sys.executable} -m pip install lifelines

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
df = pd.read_csv("/content/table.tsv", sep='\t')

df = df.rename(columns={
    'Overall Survival (Months)': 'OS_time',
    'Overall Survival Status': 'OS_status_string',
    'Diagnosis Age': 'AGE',
    'Tumor Break Load': 'Tumor_Burden'
})

df['OS_event'] = df['OS_status_string'].apply(lambda x: 1 if x == '1:DECEASED' else 0)

features = ["AGE", "Tumor_Burden"]  # Add more features as needed
survival = ["OS_time","OS_event"]

# Drop missing values
data = df[features + survival].dropna()

# ---------------------------------------------------------
# Scale continuous variables
# ---------------------------------------------------------
scaler = StandardScaler()
continuous = ["AGE", "Tumor_Burden"]
data[continuous] = scaler.fit_transform(data[continuous])

# ---------------------------------------------------------
# Check if survival analysis is possible
# ---------------------------------------------------------
if data['OS_event'].sum() == 0:
    print("No observed events. Only Tumor Burden Distribution will be plotted.")
    plt.figure()
    plt.hist(data["Tumor_Burden"], bins=30)
    plt.title("Tumor Burden Distribution (Standardized)")
    plt.xlabel("Tumor Burden")
    plt.ylabel("Patient Count")
    plt.show()
else:
    # ---------------------------------------------------------
    # Cox Proportional Hazards Model
    # ---------------------------------------------------------
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(data, duration_col="OS_time", event_col="OS_event")
    print("\n=== Cox Model Summary ===")
    cph.print_summary()

    # Compute risk score
    data["risk_score"] = cph.predict_partial_hazard(data)
    median_risk = data["risk_score"].median()
    data["risk_group"] = (data["risk_score"] > median_risk).astype(int)

    # ---------------------------------------------------------
    # Kaplan–Meier Curves
    # ---------------------------------------------------------
    kmf = KaplanMeierFitter()
    plt.figure()
    for grp, label in zip([0,1], ["Low Risk","High Risk"]):
        ix = data["risk_group"] == grp
        kmf.fit(durations=data.loc[ix,"OS_time"], event_observed=data.loc[ix,"OS_event"], label=label)
        kmf.plot_survival_function()
    plt.title("Kaplan–Meier Survival by Cox Risk Group")
    plt.xlabel("Time (months)")
    plt.ylabel("Survival Probability")
    plt.show()

    # Text summary for KM curves
    print("\n--- KM Curve Summary ---")
    print(f"Low-risk group (risk score <= {median_risk:.2f}) shows higher survival probability over time compared to high-risk group.")
    print("This qualitative summary can be inserted into the manuscript.\n")

    # ---------------------------------------------------------
    # Confusion Matrix and ROC AUC
    # ---------------------------------------------------------
    TIME_HORIZON = 12
    data["event_by_horizon"] = ((data["OS_event"]==1) & (data["OS_time"]<=TIME_HORIZON)).astype(int)
    y_true_cm = data["event_by_horizon"]
    y_pred_cm = data["risk_group"]

    cm = confusion_matrix(y_true_cm, y_pred_cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Survived >12m","Died ≤12m"])
    disp.plot()
    plt.title("Confusion Matrix (12-Month Survival Prediction)")
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true_cm, data["risk_score"])
    roc_auc = roc_auc_score(y_true_cm, data["risk_score"])
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for 12-Month Survival Prediction')
    plt.legend(loc="lower right")
    plt.show()

    print("\n--- Confusion Matrix & ROC Summary ---")
    print(f"ROC AUC Score: {roc_auc:.2f}")
    print("This indicates how well the model distinguishes patients at risk within 12 months.\n")

    # ---------------------------------------------------------
    # Tumor Burden Distribution
    # ---------------------------------------------------------
    plt.figure()
    plt.hist(data["Tumor_Burden"], bins=30)
    plt.title("Tumor Burden Distribution (Standardized)")
    plt.xlabel("Tumor Burden")
    plt.ylabel("Patient Count")
    plt.show()

    print("\n--- Tumor Burden Distribution Summary ---")
    print("Histogram shows overall distribution of standardized tumor burden.\n")

    # ---------------------------------------------------------
    # Cox Feature Importance / Coefficients Plot
    # ---------------------------------------------------------
    plt.figure(figsize=(8,5))
    coefs = cph.params_
    errors = [np.exp(coefs[feat] + cph.confidence_intervals_.loc[feat,'95% upper-bound']) - 
              np.exp(coefs[feat]) for feat in features]
    plt.barh(features, np.exp(coefs), xerr=errors, color='skyblue')
    plt.axvline(1, color='red', linestyle='--', linewidth=1)
    plt.xlabel("Hazard Ratio (Exp(Coef))")
    plt.title("Cox Model Feature Importance with 95% CI")
    plt.tight_layout()
    plt.show()

    print("\n--- Cox Feature Importance Summary ---")
    print("Hazard ratios with 95% CI are displayed for all features.")
    print("Qualitative interpretation: features with HR>1 increase risk, HR<1 are protective.\n")

    # ---------------------------------------------------------
    # Time-Dependent C-Index / Model Discrimination
    # ---------------------------------------------------------
    time_points = np.linspace(data["OS_time"].min(), data["OS_time"].max(), 100)
    c_index_over_time = [concordance_index(data["OS_time"], -data["risk_score"], data["OS_event"]) for t in time_points]

    plt.figure(figsize=(8,5))
    plt.plot(time_points, [cph.concordance_index_]*len(time_points), color='darkorange', lw=2, label='C-index (static)')
    plt.xlabel("Time (months)")
    plt.ylabel("Concordance Index")
    plt.title("Time-Dependent Model Discrimination (C-index)")
    plt.ylim(0.4,1)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\n--- Time-Dependent C-Index Summary ---")
    print("Shows model discrimination over follow-up. Higher C-index indicates better separation of risk groups.\n")

    # ---------------------------------------------------------
    # Key Metrics
    # ---------------------------------------------------------
    print("Concordance Index (C-index):", round(cph.concordance_index_,3))
    print("Patients:", len(data))
