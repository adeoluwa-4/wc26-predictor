## 📊 Overview

This app lets you pick two Euro 2024 teams and predicts the match result (win, loss, or draw).  
Instead of asking you for stats like expected goals, it calculates them automatically from past data.

The model uses:
- **Team names** and historical matchup data
- **Auto-predicted xG (expected goals)** from a regression model
- A trained **Random Forest classifier** to predict the final outcome
