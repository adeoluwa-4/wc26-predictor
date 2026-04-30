"""WC26 visual theme helpers for Streamlit pages."""

from __future__ import annotations

import plotly.io as pio
import streamlit as st


def apply_wc26_theme() -> None:
    """Apply a WC26-inspired theme with stable contrast across device modes."""
    pio.templates.default = "plotly_dark"
    st.markdown(
        """
<style>
:root {
  --wc-red: #d90404;
  --wc-purple: #5a11d4;
  --wc-lime: #b6e603;
  --wc-deep: #070b18;
  --wc-card: rgba(9, 14, 30, 0.88);
  --wc-panel: rgba(6, 10, 22, 0.90);
  --wc-border: rgba(255, 255, 255, 0.20);
  --wc-text: #eef2ff;
  --wc-muted: #b7c1de;
  --wc-input: rgba(8, 14, 32, 0.95);
}

.stApp {
  color: var(--wc-text);
  font-family: "Segoe UI", "Inter", "Segoe UI Emoji", "Segoe UI Symbol", "Apple Color Emoji", "Noto Color Emoji", sans-serif;
  background:
    radial-gradient(circle at 22% 18%, rgba(90, 17, 212, 0.70) 0%, rgba(90, 17, 212, 0.0) 38%),
    radial-gradient(circle at 72% 24%, rgba(217, 4, 4, 0.68) 0%, rgba(217, 4, 4, 0.0) 42%),
    radial-gradient(circle at 88% 70%, rgba(182, 230, 3, 0.44) 0%, rgba(182, 230, 3, 0.0) 36%),
    linear-gradient(140deg, #090d1f 0%, #0f132a 38%, #0b0f1d 100%);
  background-attachment: fixed;
}

div[data-testid="stHeader"] {
  background: transparent;
}

div[data-testid="stAppViewContainer"] > .main,
section.main,
section.main > div {
  background: transparent;
}

div[data-testid="stSidebar"] {
  background: var(--wc-panel);
  border-right: 1px solid var(--wc-border);
}

div[data-testid="stSidebarNav"] {
  background: rgba(8, 13, 28, 0.86);
  border: 1px solid var(--wc-border);
  border-radius: 12px;
  padding: 8px 8px 4px 8px;
  margin-bottom: 12px;
}

div[data-testid="stSidebarNav"] a {
  color: var(--wc-text) !important;
  border-radius: 10px;
}

div[data-testid="stSidebarNav"] a:hover {
  background: rgba(255, 255, 255, 0.10) !important;
}

div[data-testid="stMetric"] {
  background: var(--wc-card);
  border: 1px solid var(--wc-border);
  border-radius: 14px;
  padding: 10px 14px;
}

div[data-testid="stDataFrame"],
div[data-testid="stPlotlyChart"],
div[data-testid="stImage"] {
  background: var(--wc-card);
  border: 1px solid var(--wc-border);
  border-radius: 14px;
  padding: 4px;
}

/* Keep text readable if host/device is in light mode */
div[data-testid="stMarkdownContainer"],
div[data-testid="stText"],
label,
p,
span,
h1, h2, h3, h4, h5, h6 {
  color: var(--wc-text);
}

.stCaption {
  color: var(--wc-muted) !important;
}

/* Input controls and dropdowns */
div[data-baseweb="select"] > div,
div[data-baseweb="base-input"] > div,
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
  background: var(--wc-input) !important;
  color: var(--wc-text) !important;
  border-color: var(--wc-border) !important;
}

/* Sidebar controls */
div[data-testid="stSidebar"] * {
  font-family: "Segoe UI", "Inter", "Segoe UI Emoji", "Segoe UI Symbol", "Apple Color Emoji", "Noto Color Emoji", sans-serif !important;
}

/* Top page links for mobile navigation */
div[data-testid="stPageLink"] > a {
  background: rgba(9, 14, 30, 0.82);
  border: 1px solid var(--wc-border);
  border-radius: 10px;
  color: var(--wc-text) !important;
}
div[data-testid="stPageLink"] > a:hover {
  background: rgba(255, 255, 255, 0.10);
}

/* Plotly text should be able to render flag emoji on Windows */
.js-plotly-plot .plotly text {
  font-family: "Segoe UI Emoji", "Segoe UI Symbol", "Segoe UI", "Apple Color Emoji", "Noto Color Emoji", sans-serif !important;
}

.wc-photo-card {
  background: var(--wc-card);
  border: 1px solid var(--wc-border);
  border-radius: 14px;
  padding: 8px;
}
</style>
        """,
        unsafe_allow_html=True,
    )
