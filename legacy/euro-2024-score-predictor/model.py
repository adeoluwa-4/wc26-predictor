# model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("data/matches.csv")

# Add outcome column
def get_outcome(row):
    if row["home_goals"] > row["away_goals"]:
        return 1
    elif row["home_goals"] < row["away_goals"]:
        return 2
    else:
        return 0

df["outcome"] = df.apply(get_outcome, axis=1)

# Train xG regression models
team_features = ["home_team", "away_team"]
df_encoded = pd.get_dummies(df[team_features])

X_reg = df_encoded
y_home_xg = df["Home Expected goals(xG)"]
y_away_xg = df["Away Expected goals(xG)"]

home_xg_model = RandomForestRegressor(n_estimators=100, random_state=42)
home_xg_model.fit(X_reg, y_home_xg)

away_xg_model = RandomForestRegressor(n_estimators=100, random_state=42)
away_xg_model.fit(X_reg, y_away_xg)

# Predict xG
df["home_xg_pred"] = home_xg_model.predict(X_reg)
df["away_xg_pred"] = away_xg_model.predict(X_reg)

# Train final outcome model on predicted xG
X_clf = df[["home_xg_pred", "away_xg_pred"]]
y_clf = df["outcome"]

X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Outcome model accuracy (with predicted xG): {acc:.2f}")

# Save all models
joblib.dump(home_xg_model, "models/home_xg_model.pkl")
joblib.dump(away_xg_model, "models/away_xg_model.pkl")
joblib.dump(clf, "models/outcome_model.pkl")
