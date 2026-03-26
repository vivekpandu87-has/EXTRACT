import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils import (
    MOOD_COLORS, INTEREST_COLORS, PRIMARY, SECONDARY, ACCENT,
    INCOME_ORDER, AGE_ORDER, FREQ_ORDER, LAST_BUY_ORDER,
    build_segment_profile, PSM_MIDPOINTS,
)

# ─── palette helpers ────────────────────────────────────────────────────────
PASTEL  = px.colors.qualitative.Pastel
BOLD    = px.colors.qualitative.Bold
SET2    = px.colors.qualitative.Set2
PLOTLY  = px.colors.qualitative.Plotly

CHART_BG   = "rgba(0,0,0,0)"
PAPER_BG   = "rgba(0,0,0,0)"
FONT_COLOR = "#e8e0f0"
GRID_COLOR = "rgba(168,85,247,0.12)"

def _layout(fig, title="", height=420):
    fig.update_layout(
        title=title,
        height=height,
        plot_bgcolor=CHART_BG,
        paper_bgcolor=PAPER_BG,
        font=dict(color=FONT_COLOR, size=12),
        title_font=dict(size=15, color=PRIMARY),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=30, r=20, t=50, b=30),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    )
    return fig


def _kpi(col, label, value, delta=None):
    col.markdown(f"""
    <div style="background:linear-gradient(135deg,#1a1a2e,#2a1a3e);
                border:1px solid {PRIMARY}33; border-radius:12px;
                padding:18px 16px; text-align:center; margin-bottom:8px;">
        <div style="font-size:24px; font-weight:700; color:{PRIMARY};">{value}</div>
        <div style="font-size:12px; color:#b0a0c0; margin-top:4px;">{label}</div>
        {"<div style='font-size:11px;color:#4CAF50;margin-top:2px;'>"+delta+"</div>" if delta else ""}
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
def run_eda(df: pd.DataFrame):
    st.markdown(f"""
    <h2 style="color:{PRIMARY};margin-bottom:4px;">📊 Exploratory Data Analysis</h2>
    <p style="color:#b0a0c0;margin-top:0;">Deep-dive into MoodCart survey responses — demographics,
    behaviours, mood patterns, spend & market sizing.</p>
    """, unsafe_allow_html=True)

    tabs = st.tabs([
        "🔍 Overview", "👥 Demographics", "🛒 Shopping Behaviour",
        "🧠 Mood & Emotions", "💰 Spend & PSM", "📦 Products & Features",
        "🚧 Barriers & Drivers",
    ])

    # ── 0 OVERVIEW ────────────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("Dataset Snapshot")
        df_num = df.copy()
        df_num["Monthly_Spend"] = pd.to_numeric(df_num["Monthly_Spend"], errors="coerce")

        c1, c2, c3, c4, c5 = st.columns(5)
        _kpi(c1, "Total Respondents", f"{len(df):,}")
        _kpi(c2, "Columns", f"{df.shape[1]}")
        _kpi(c3, "Avg Monthly Spend", f"₹{df_num['Monthly_Spend'].mean():,.0f}")
        _kpi(c4, "Interested in MoodCart",
             f"{(df['Interest_in_MoodCart']=='Yes').sum():,}",
             f"{(df['Interest_in_MoodCart']=='Yes').mean()*100:.1f}% of total")
        _kpi(c5, "Unique Moods",
             f"{df['Mood'].nunique()}" if 'Mood' in df.columns else "—")

        st.markdown("---")
        st.subheader("📋 Raw Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("🎯 Interest in MoodCart")
        counts = df["Interest_in_MoodCart"].value_counts().reset_index()
        counts.columns = ["Interest", "Count"]
        counts["Pct"] = (counts["Count"] / counts["Count"].sum() * 100).round(1)

        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.bar(counts, x="Interest", y="Count",
                         color="Interest", text=counts["Pct"].astype(str) + "%",
                         color_discrete_map=INTEREST_COLORS)
            fig.update_traces(textposition="outside")
            st.plotly_chart(_layout(fig, "Interest Distribution"), use_container_width=True)
        with col_b:
            fig2 = px.pie(counts, names="Interest", values="Count",
                          hole=0.5,
                          color="Interest", color_discrete_map=INTEREST_COLORS)
            fig2.update_traces(textinfo="percent+label")
            st.plotly_chart(_layout(fig2, "Interest Share"), use_container_width=True)

        # missing values
        st.subheader("🔎 Missing Values")
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if missing.empty:
            st.success("✅ No missing values found.")
        else:
            m_df = missing.reset_index()
            m_df.columns = ["Column", "Missing"]
            m_df["Pct"] = (m_df["Missing"] / len(df) * 100).round(2)
            st.dataframe(m_df, use_container_width=True)

    # ── 1 DEMOGRAPHICS ────────────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("👥 Respondent Demographics")
        col1, col2 = st.columns(2)

        with col1:
            if "Age" in df.columns:
                age_counts = df["Age"].value_counts().reindex(AGE_ORDER).reset_index()
                age_counts.columns = ["Age", "Count"]
                fig = px.bar(age_counts, x="Age", y="Count", color="Age",
                             color_discrete_sequence=BOLD, text="Count")
                fig.update_traces(textposition="outside")
                st.plotly_chart(_layout(fig, "Age Group Distribution"), use_container_width=True)

            if "Occupation" in df.columns:
                occ = df["Occupation"].value_counts().reset_index()
                occ.columns = ["Occupation", "Count"]
                fig = px.pie(occ, names="Occupation", values="Count",
                             hole=0.4, color_discrete_sequence=PASTEL)
                st.plotly_chart(_layout(fig, "Occupation Split"), use_container_width=True)

        with col2:
            if "Gender" in df.columns:
                gen = df["Gender"].value_counts().reset_index()
                gen.columns = ["Gender", "Count"]
                fig = px.pie(gen, names="Gender", values="Count",
                             hole=0.45, color_discrete_sequence=[PRIMARY, ACCENT, "#26C6DA", "#FF9800"])
                st.plotly_chart(_layout(fig, "Gender Distribution"), use_container_width=True)

            if "City_Tier" in df.columns:
                tier = df["City_Tier"].value_counts().reset_index()
                tier.columns = ["City_Tier", "Count"]
                fig = px.bar(tier, x="City_Tier", y="Count", color="City_Tier",
                             color_discrete_sequence=SET2, text="Count")
                fig.update_traces(textposition="outside")
                st.plotly_chart(_layout(fig, "City Tier Distribution"), use_container_width=True)

        if "Income" in df.columns:
            st.subheader("💵 Income Distribution")
            inc = df["Income"].value_counts().reindex(INCOME_ORDER).reset_index()
            inc.columns = ["Income", "Count"]
            fig = px.bar(inc, x="Income", y="Count", color="Income",
                         color_discrete_sequence=BOLD, text="Count")
            fig.update_traces(textposition="outside")
            st.plotly_chart(_layout(fig, "Income Bracket Distribution"), use_container_width=True)

        # cross-tab: age × interest
        if "Age" in df.columns and "Interest_in_MoodCart" in df.columns:
            st.subheader("📊 Age × Interest in MoodCart")
            ct = pd.crosstab(df["Age"], df["Interest_in_MoodCart"])
            ct = ct.reindex(AGE_ORDER).fillna(0)
            ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
            melted = ct_pct.reset_index().melt(id_vars="Age")
            melted["text_val"] = melted["value"].round(1).astype(str) + "%"
            fig = px.bar(melted,
                         x="Age", y="value", color="variable",
                         barmode="stack", text="text_val",
                         color_discrete_map=INTEREST_COLORS,
                         labels={"value": "%", "variable": "Interest"})
            st.plotly_chart(_layout(fig, "Interest % by Age Group"), use_container_width=True)

        # cross-tab: income × interest
        if "Income" in df.columns and "Interest_in_MoodCart" in df.columns:
            st.subheader("📊 Income × Interest in MoodCart")
            ct2 = pd.crosstab(df["Income"], df["Interest_in_MoodCart"])
            ct2 = ct2.reindex(INCOME_ORDER).fillna(0)
            ct2_pct = ct2.div(ct2.sum(axis=1), axis=0) * 100
            melted2 = ct2_pct.reset_index().melt(id_vars="Income")
            melted2["text_val"] = melted2["value"].round(1).astype(str) + "%"
            fig = px.bar(melted2,
                         x="Income", y="value", color="variable",
                         barmode="stack", text="text_val",
                         color_discrete_map=INTEREST_COLORS,
                         labels={"value": "%", "variable": "Interest"})
            st.plotly_chart(_layout(fig, "Interest % by Income Bracket"), use_container_width=True)

    # ── 2 SHOPPING BEHAVIOUR ──────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("🛒 Shopping Behaviour Patterns")
        col1, col2 = st.columns(2)

        with col1:
            if "Shopping_Frequency" in df.columns:
                freq = df["Shopping_Frequency"].value_counts().reindex(FREQ_ORDER).reset_index()
                freq.columns = ["Frequency", "Count"]
                fig = px.bar(freq, x="Frequency", y="Count",
                             color="Frequency", text="Count",
                             color_discrete_sequence=BOLD)
                fig.update_traces(textposition="outside")
                st.plotly_chart(_layout(fig, "Shopping Frequency"), use_container_width=True)

            if "Habit_Type" in df.columns:
                habit = df["Habit_Type"].value_counts().reset_index()
                habit.columns = ["Habit", "Count"]
                fig = px.pie(habit, names="Habit", values="Count",
                             hole=0.4, color_discrete_sequence=SET2)
                st.plotly_chart(_layout(fig, "Habit Type"), use_container_width=True)

        with col2:
            if "Last_Purchase" in df.columns:
                lp = df["Last_Purchase"].value_counts().reindex(LAST_BUY_ORDER).reset_index()
                lp.columns = ["Last_Purchase", "Count"]
                fig = px.bar(lp, x="Last_Purchase", y="Count",
                             color="Last_Purchase", text="Count",
                             color_discrete_sequence=PASTEL)
                fig.update_traces(textposition="outside")
                st.plotly_chart(_layout(fig, "Last Purchase Recency"), use_container_width=True)

            if "Cart_Abandonment" in df.columns:
                ca = df["Cart_Abandonment"].value_counts().reset_index()
                ca.columns = ["Cart_Abandonment", "Count"]
                fig = px.pie(ca, names="Cart_Abandonment", values="Count",
                             hole=0.45,
                             color_discrete_sequence=[PRIMARY, ACCENT, "#FF9800", "#4CAF50"])
                st.plotly_chart(_layout(fig, "Cart Abandonment Frequency"), use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            if "Browsing_Time" in df.columns:
                bt = df["Browsing_Time"].value_counts().reset_index()
                bt.columns = ["Browsing_Time", "Count"]
                fig = px.bar(bt, x="Browsing_Time", y="Count",
                             color="Browsing_Time", text="Count",
                             color_discrete_sequence=BOLD)
                fig.update_traces(textposition="outside")
                st.plotly_chart(_layout(fig, "Browsing Time per Session"), use_container_width=True)

        with col4:
            if "Shopping_Time" in df.columns:
                sht = df["Shopping_Time"].value_counts().reset_index()
                sht.columns = ["Shopping_Time", "Count"]
                fig = px.pie(sht, names="Shopping_Time", values="Count",
                             hole=0.4, color_discrete_sequence=SET2)
                st.plotly_chart(_layout(fig, "Preferred Shopping Time"), use_container_width=True)

        if "Purchase_Influence" in df.columns:
            st.subheader("📣 Purchase Influence Factors")
            infl = df["Purchase_Influence"].value_counts().reset_index()
            infl.columns = ["Influence", "Count"]
            fig = px.bar(infl.sort_values("Count"), x="Count", y="Influence",
                         orientation="h", color="Count",
                         color_continuous_scale=["#2a1a3e", PRIMARY, ACCENT])
            st.plotly_chart(_layout(fig, "What Influences Purchase Decisions?", 380), use_container_width=True)

        if "Decision_Style" in df.columns:
            st.subheader("🧩 Decision Style Distribution")
            ds = df["Decision_Style"].value_counts().reset_index()
            ds.columns = ["Style", "Count"]
            fig = px.pie(ds, names="Style", values="Count",
                         hole=0.45, color_discrete_sequence=BOLD)
            st.plotly_chart(_layout(fig, "Decision Style"), use_container_width=True)

    # ── 3 MOOD & EMOTIONS ─────────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("🧠 Mood & Emotional Patterns")

        if "Mood" in df.columns:
            col1, col2 = st.columns(2)
            mood_counts = df["Mood"].value_counts().reset_index()
            mood_counts.columns = ["Mood", "Count"]
            mood_counts["color"] = mood_counts["Mood"].map(MOOD_COLORS)

            with col1:
                fig = px.bar(mood_counts, x="Mood", y="Count",
                             color="Mood",
                             color_discrete_map=MOOD_COLORS, text="Count")
                fig.update_traces(textposition="outside")
                st.plotly_chart(_layout(fig, "Primary Mood Distribution"), use_container_width=True)

            with col2:
                fig = px.pie(mood_counts, names="Mood", values="Count",
                             hole=0.45, color="Mood",
                             color_discrete_map=MOOD_COLORS)
                st.plotly_chart(_layout(fig, "Mood Share"), use_container_width=True)

        # mood vs interest heatmap
        if "Mood" in df.columns and "Interest_in_MoodCart" in df.columns:
            st.subheader("🧩 Mood × Interest in MoodCart")
            ct = pd.crosstab(df["Mood"], df["Interest_in_MoodCart"])
            ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
            fig = px.imshow(ct_pct.round(1),
                            color_continuous_scale=["#0d0d1a", SECONDARY, PRIMARY, ACCENT],
                            text_auto=True,
                            labels=dict(color="% Interest"))
            st.plotly_chart(_layout(fig, "Mood vs Interest Heatmap (% row)", 400), use_container_width=True)

        # mood vs avg spend
        if "Mood" in df.columns and "Monthly_Spend" in df.columns:
            st.subheader("💸 Avg Monthly Spend by Mood")
            df3 = df.copy()
            df3["Monthly_Spend"] = pd.to_numeric(df3["Monthly_Spend"], errors="coerce")
            ms = df3.groupby("Mood")["Monthly_Spend"].mean().reset_index()
            ms.columns = ["Mood", "Avg_Spend"]
            ms = ms.sort_values("Avg_Spend", ascending=False)
            fig = px.bar(ms, x="Mood", y="Avg_Spend",
                         color="Mood", text=ms["Avg_Spend"].round(0),
                         color_discrete_map=MOOD_COLORS)
            fig.update_traces(texttemplate="₹%{text}", textposition="outside")
            st.plotly_chart(_layout(fig, "Average Spend per Mood State"), use_container_width=True)

        # emotional frequency
        if "Emotional_Frequency" in df.columns:
            st.subheader("⚡ Emotional Shopping Frequency")
            ef = df["Emotional_Frequency"].value_counts().reset_index()
            ef.columns = ["Frequency", "Count"]
            fig = px.bar(ef, x="Frequency", y="Count",
                         color="Frequency", text="Count",
                         color_discrete_sequence=BOLD)
            fig.update_traces(textposition="outside")
            st.plotly_chart(_layout(fig, "How Often Emotions Drive Shopping"), use_container_width=True)

        # impulse behaviour
        col5, col6 = st.columns(2)
        with col5:
            if "Impulse_Behavior" in df.columns:
                ib = df["Impulse_Behavior"].value_counts().reset_index()
                ib.columns = ["Impulse", "Count"]
                fig = px.pie(ib, names="Impulse", values="Count",
                             hole=0.4,
                             color_discrete_sequence=[PRIMARY, ACCENT, "#FF9800", "#4CAF50"])
                st.plotly_chart(_layout(fig, "Impulse Buying Behaviour"), use_container_width=True)

        with col6:
            if "Mood_Impact" in df.columns:
                mi = df["Mood_Impact"].value_counts().reset_index()
                mi.columns = ["Mood_Impact", "Count"]
                fig = px.bar(mi, x="Mood_Impact", y="Count",
                             color="Mood_Impact", text="Count",
                             color_discrete_sequence=SET2)
                fig.update_traces(textposition="outside")
                st.plotly_chart(_layout(fig, "Does Mood Impact Shopping?"), use_container_width=True)

        # post purchase feeling
        if "Post_Purchase_Feeling" in df.columns:
            st.subheader("😌 Post-Purchase Feelings")
            ppf = df["Post_Purchase_Feeling"].value_counts().reset_index()
            ppf.columns = ["Feeling", "Count"]
            fig = px.bar(ppf.sort_values("Count", ascending=True),
                         x="Count", y="Feeling", orientation="h",
                         color="Count",
                         color_continuous_scale=["#2a1a3e", PRIMARY, ACCENT])
            st.plotly_chart(_layout(fig, "Post-Purchase Emotion Distribution", 380), use_container_width=True)

    # ── 4 SPEND & PSM ─────────────────────────────────────────────────────────
    with tabs[4]:
        st.subheader("💰 Spend Analysis")
        df_s = df.copy()
        df_s["Monthly_Spend"] = pd.to_numeric(df_s["Monthly_Spend"], errors="coerce")

        c1, c2, c3, c4 = st.columns(4)
        _kpi(c1, "Mean Spend",   f"₹{df_s['Monthly_Spend'].mean():,.0f}")
        _kpi(c2, "Median Spend", f"₹{df_s['Monthly_Spend'].median():,.0f}")
        _kpi(c3, "Max Spend",    f"₹{df_s['Monthly_Spend'].max():,.0f}")
        _kpi(c4, "Std Dev",      f"₹{df_s['Monthly_Spend'].std():,.0f}")

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df_s, x="Monthly_Spend",
                               nbins=40,
                               color_discrete_sequence=[PRIMARY])
            fig.update_traces(marker_line_color=ACCENT, marker_line_width=0.5)
            st.plotly_chart(_layout(fig, "Monthly Spend Distribution"), use_container_width=True)

        with col2:
            if "Income" in df.columns:
                fig = px.box(df_s, x="Income", y="Monthly_Spend",
                             color="Income",
                             category_orders={"Income": INCOME_ORDER},
                             color_discrete_sequence=BOLD)
                st.plotly_chart(_layout(fig, "Spend by Income Bracket"), use_container_width=True)

        if "Age" in df.columns:
            fig = px.box(df_s, x="Age", y="Monthly_Spend",
                         color="Age",
                         category_orders={"Age": AGE_ORDER},
                         color_discrete_sequence=BOLD)
            st.plotly_chart(_layout(fig, "Spend by Age Group"), use_container_width=True)

        if "Willingness_To_Spend_More" in df.columns:
            st.subheader("📈 Willingness to Spend More")
            wts = df["Willingness_To_Spend_More"].value_counts().reset_index()
            wts.columns = ["Willingness", "Count"]
            wts["Pct"] = (wts["Count"] / wts["Count"].sum() * 100).round(1)
            col3, col4 = st.columns(2)
            with col3:
                fig = px.bar(wts, x="Willingness", y="Count",
                             color="Willingness", text=wts["Pct"].astype(str) + "%",
                             color_discrete_sequence=[PRIMARY, ACCENT, "#FF9800"])
                fig.update_traces(textposition="outside")
                st.plotly_chart(_layout(fig, "Willingness to Spend More on Personalized Reco"), use_container_width=True)
            with col4:
                fig2 = px.pie(wts, names="Willingness", values="Count", hole=0.45,
                              color_discrete_sequence=[PRIMARY, ACCENT, "#FF9800"])
                st.plotly_chart(_layout(fig2, "Spend Willingness Share"), use_container_width=True)

        # WTP vs mood
        if "Mood" in df.columns and "Willingness_To_Spend_More" in df.columns:
            st.subheader("🧠 Willingness to Spend More × Mood")
            ct = pd.crosstab(df["Mood"], df["Willingness_To_Spend_More"])
            ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
            fig = px.imshow(ct_pct.round(1),
                            color_continuous_scale=["#0d0d1a", SECONDARY, PRIMARY, ACCENT],
                            text_auto=True)
            st.plotly_chart(_layout(fig, "Mood vs WTP Heatmap (% row)", 380), use_container_width=True)

    # ── 5 PRODUCTS & FEATURES ─────────────────────────────────────────────────
    with tabs[5]:
        st.subheader("📦 Category & Product Preferences")

        if "Categories" in df.columns:
            cats_raw = df["Categories"].fillna("")
            all_cats: dict = {}
            for row in cats_raw:
                for c in str(row).split("|"):
                    c = c.strip()
                    if c:
                        all_cats[c] = all_cats.get(c, 0) + 1
            cats_df = pd.DataFrame(list(all_cats.items()), columns=["Category", "Count"])
            cats_df = cats_df.sort_values("Count", ascending=True)
            fig = px.bar(cats_df, x="Count", y="Category",
                         orientation="h", color="Count",
                         color_continuous_scale=["#2a1a3e", PRIMARY, ACCENT])
            st.plotly_chart(_layout(fig, "Category Preferences (multi-select)", 380), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            if "Stress_Purchases" in df.columns:
                sp_raw = df["Stress_Purchases"].fillna("")
                sp_dict: dict = {}
                for row in sp_raw:
                    for c in str(row).split("|"):
                        c = c.strip()
                        if c:
                            sp_dict[c] = sp_dict.get(c, 0) + 1
                sp_df = pd.DataFrame(list(sp_dict.items()), columns=["Category", "Count"])
                sp_df = sp_df.sort_values("Count", ascending=False)
                fig = px.bar(sp_df, x="Category", y="Count",
                             color="Category", text="Count",
                             color_discrete_sequence=BOLD)
                fig.update_traces(textposition="outside")
                st.plotly_chart(_layout(fig, "Stress Purchase Categories"), use_container_width=True)

        with col2:
            if "Shopping_Situations" in df.columns:
                sit_raw = df["Shopping_Situations"].fillna("")
                sit_dict: dict = {}
                for row in sit_raw:
                    for c in str(row).split("|"):
                        c = c.strip()
                        if c:
                            sit_dict[c] = sit_dict.get(c, 0) + 1
                sit_df = pd.DataFrame(list(sit_dict.items()), columns=["Situation", "Count"])
                sit_df = sit_df.sort_values("Count", ascending=False)
                fig = px.bar(sit_df, x="Situation", y="Count",
                             color="Situation", text="Count",
                             color_discrete_sequence=SET2)
                fig.update_traces(textposition="outside")
                st.plotly_chart(_layout(fig, "Shopping Trigger Situations"), use_container_width=True)

        if "Product_Combinations" in df.columns:
            st.subheader("🧺 Popular Product Bundle Combinations")
            pc_raw = df["Product_Combinations"].fillna("")
            pc_dict: dict = {}
            for row in pc_raw:
                for c in str(row).split("|"):
                    c = c.strip()
                    if c:
                        pc_dict[c] = pc_dict.get(c, 0) + 1
            pc_df = pd.DataFrame(list(pc_dict.items()), columns=["Bundle", "Count"])
            pc_df = pc_df.sort_values("Count", ascending=True)
            fig = px.bar(pc_df, x="Count", y="Bundle",
                         orientation="h", color="Count",
                         color_continuous_scale=["#2a1a3e", PRIMARY, ACCENT])
            st.plotly_chart(_layout(fig, "Product Bundle Frequency", 380), use_container_width=True)

        # value perception & priority
        col3, col4 = st.columns(2)
        with col3:
            if "Value_Perception" in df.columns:
                vp = df["Value_Perception"].value_counts().reset_index()
                vp.columns = ["Perception", "Count"]
                fig = px.pie(vp, names="Perception", values="Count",
                             hole=0.4, color_discrete_sequence=PASTEL)
                st.plotly_chart(_layout(fig, "Value Perception"), use_container_width=True)
        with col4:
            if "Priority" in df.columns:
                pr = df["Priority"].value_counts().reset_index()
                pr.columns = ["Priority", "Count"]
                fig = px.bar(pr.sort_values("Count"), x="Count", y="Priority",
                             orientation="h", color="Count",
                             color_continuous_scale=["#2a1a3e", PRIMARY, ACCENT])
                st.plotly_chart(_layout(fig, "Top Purchase Priority Factors", 380), use_container_width=True)

    # ── 6 BARRIERS & DRIVERS ──────────────────────────────────────────────────
    with tabs[6]:
        st.subheader("🚧 Barriers, Hesitations & AI Trust")

        col1, col2 = st.columns(2)
        with col1:
            if "Hesitation" in df.columns:
                hes = df["Hesitation"].value_counts().reset_index()
                hes.columns = ["Hesitation", "Count"]
                hes = hes.sort_values("Count")
                fig = px.bar(hes, x="Count", y="Hesitation",
                             orientation="h", color="Count",
                             color_continuous_scale=["#2a1a3e", PRIMARY, ACCENT])
                st.plotly_chart(_layout(fig, "Main Hesitation Factors", 380), use_container_width=True)

        with col2:
            if "Tradeoff" in df.columns:
                to = df["Tradeoff"].value_counts().reset_index()
                to.columns = ["Tradeoff", "Count"]
                fig = px.pie(to, names="Tradeoff", values="Count",
                             hole=0.4, color_discrete_sequence=BOLD)
                st.plotly_chart(_layout(fig, "Price vs Quality Tradeoff"), use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            if "AI_Trust" in df.columns:
                at = df["AI_Trust"].value_counts().reset_index()
                at.columns = ["AI_Trust", "Count"]
                fig = px.bar(at, x="AI_Trust", y="Count",
                             color="AI_Trust", text="Count",
                             color_discrete_sequence=[PRIMARY, ACCENT, "#FF9800", "#4CAF50"])
                fig.update_traces(textposition="outside")
                st.plotly_chart(_layout(fig, "AI Trust Level"), use_container_width=True)

        with col4:
            if "Privacy_Comfort" in df.columns:
                pc = df["Privacy_Comfort"].value_counts().reset_index()
                pc.columns = ["Privacy_Comfort", "Count"]
                fig = px.pie(pc, names="Privacy_Comfort", values="Count",
                             hole=0.4,
                             color_discrete_sequence=[PRIMARY, SECONDARY, ACCENT, "#FF9800"])
                st.plotly_chart(_layout(fig, "Privacy Comfort Level"), use_container_width=True)

        col5, col6 = st.columns(2)
        with col5:
            if "Data_Concern" in df.columns:
                dc = df["Data_Concern"].value_counts().reset_index()
                dc.columns = ["Data_Concern", "Count"]
                dc = dc.sort_values("Count")
                fig = px.bar(dc, x="Count", y="Data_Concern",
                             orientation="h", color="Count",
                             color_continuous_scale=["#2a1a3e", PRIMARY, ACCENT])
                st.plotly_chart(_layout(fig, "Data Concern Reasons", 350), use_container_width=True)

        with col6:
            if "Pre_Purchase_Action" in df.columns:
                ppa = df["Pre_Purchase_Action"].value_counts().reset_index()
                ppa.columns = ["Action", "Count"]
                fig = px.pie(ppa, names="Action", values="Count",
                             hole=0.4, color_discrete_sequence=SET2)
                st.plotly_chart(_layout(fig, "Pre-Purchase Actions"), use_container_width=True)
