import pandas as pd
import numpy as np

print(" Starting Feature Engineering...")

# =========================
# 1. Load Data
# =========================
df = pd.read_csv("jobs.csv")

# =========================
# 2. Financial Features
# =========================

df["total_compensation"] = df["salary_usd"] + df["bonus_usd"]

df["net_income_after_tax"] = df["total_compensation"] * (
    1 - (df["tax_rate_percent"] / 100)
)

df["salary_per_hour"] = df["salary_usd"] / (df["weekly_hours"] * 52)

# Adjusted salary (real purchasing power)
df["real_salary"] = df["salary_usd"] / df["cost_of_living_index"]

# =========================
# 3. AI Classification
# =========================

df["role_category"] = np.where(
    df["ai_specialization"].isna() | (df["ai_specialization"] == ""),
    "Traditional Tech",
    "AI Role",
)

# =========================
# 4. Experience Segmentation
# =========================

df["experience_bucket"] = pd.cut(
    df["experience_years"],
    bins=[0, 2, 5, 10, 20, 50],
    labels=["Entry", "Junior", "Mid", "Senior", "Expert"],
)

# =========================
# 5. Salary Segmentation
# =========================

df["salary_level"] = pd.qcut(
    df["salary_usd"], 4, labels=["Low", "Medium", "High", "Very High"]
)

# =========================
# 6. Risk & Safety Features
# =========================

df["net_safety_score"] = df["job_security_score"] - df["automation_risk"]

df["risk_category"] = pd.cut(
    df["automation_risk"],
    bins=[0, 30, 60, 100],
    labels=["Low Risk", "Medium Risk", "High Risk"],
)

# =========================
# 7. Work Environment Features
# =========================

df["work_intensity"] = df["weekly_hours"] / df["work_life_balance_score"]

df["is_remote"] = np.where(df["work_mode"] == "Remote", 1, 0)

# =========================
# 8. Growth Features
# =========================

df["growth_potential"] = (
    df["career_growth_score"] + df["promotion_speed"] + df["skill_demand_score"]
) / 3

# =========================
# 9. Market Demand Features
# =========================

df["demand_pressure"] = df["job_openings"] / df["hiring_difficulty_score"]

# =========================
# 10. Company Strength
# =========================

df["company_strength"] = (
    df["company_rating"] + df["company_funding_billion"] + df["ai_adoption_score"]
) / 3

# =========================
# 11. Satisfaction Insights
# =========================

df["satisfaction_gap"] = df["employee_satisfaction"] - df["work_life_balance_score"]

# =========================
# 12. Save Clean Data
# =========================

df.to_csv("AI_Jobs_Featured.csv", index=False)

print(" Feature Engineering Completed Successfully!")
