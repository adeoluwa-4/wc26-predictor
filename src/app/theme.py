"""WC26 visual theme helpers for Streamlit pages."""

from __future__ import annotations

import streamlit as st


def apply_wc26_theme() -> None:
    """Apply a WC26-inspired colorful background/theme."""
    st.markdown(
        """
<style>
:root {
  --wc-red: #d90404;
  --wc-purple: #5a11d4;
  --wc-lime: #b6e603;
  --wc-deep: #070b18;
  --wc-card: rgba(10, 16, 34, 0.72);
  --wc-border: rgba(255, 255, 255, 0.14);
}

.stApp {
  background:
    radial-gradient(circle at 22% 18%, rgba(90, 17, 212, 0.85) 0%, rgba(90, 17, 212, 0.0) 36%),
    radial-gradient(circle at 72% 24%, rgba(217, 4, 4, 0.86) 0%, rgba(217, 4, 4, 0.0) 40%),
    radial-gradient(circle at 88% 70%, rgba(182, 230, 3, 0.62) 0%, rgba(182, 230, 3, 0.0) 34%),
    linear-gradient(140deg, #090d1f 0%, #0f132a 38%, #0b0f1d 100%);
  background-attachment: fixed;
}

div[data-testid="stHeader"] {
  background: transparent;
}

div[data-testid="stSidebar"] {
  background: rgba(6, 10, 22, 0.82);
  border-right: 1px solid var(--wc-border);
}

/* Force first nav item label to "Overview" as a fallback when pages.toml label is ignored. */
div[data-testid="stSidebarNav"] ul li:first-child a {
  position: relative;
}
div[data-testid="stSidebarNav"] ul li:first-child a p {
  visibility: hidden;
}
div[data-testid="stSidebarNav"] ul li:first-child a::after {
  content: "Overview";
  position: absolute;
  left: 14px;
  top: 50%;
  transform: translateY(-50%);
  color: rgb(250, 250, 250);
  font-size: 1.05rem;
  font-weight: 600;
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
  border-radius: 14px;
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
