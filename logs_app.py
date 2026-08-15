"""
Chatbot Monitoring Application

This application is intentionally separate from the
main chatbot UI.

Run:

    streamlit run logs_app.py --server.port 8502
"""

import streamlit as st

from logging_service.json_logger import read_logs


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Chatbot Logs",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "📊 Chatbot Interaction Logs"
)

st.caption(
    "Monitoring and interaction history"
)


# ============================================================
# LOAD LOGS
# ============================================================

logs = read_logs()


# ============================================================
# EMPTY STATE
# ============================================================

if not logs:

    st.info(
        "No chatbot interactions have been logged yet."
    )

    st.stop()


# ============================================================
# SUMMARY
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Interactions",
        len(logs)
    )

with col2:

    st.metric(
        "Users",
        len({
            log.get("user_id")
            for log in logs
        })
    )

with col3:

    st.metric(
        "Sessions",
        len({
            log.get("session_id")
            for log in logs
        })
    )

with col4:

    st.metric(
        "Total Tokens",
        sum(
            log.get(
                "total_tokens",
                0
            )
            for log in logs
        )
    )


st.divider()


# ============================================================
# FILTERS
# ============================================================

col1, col2 = st.columns(2)

with col1:

    routes = [
        "All"
    ] + sorted({
        log.get("route", "")
        for log in logs
    })

    selected_route = st.selectbox(
        "Route",
        routes
    )


with col2:

    users = [
        "All"
    ] + sorted({
        log.get("user_id", "")
        for log in logs
    })

    selected_user = st.selectbox(
        "User",
        users
    )


# ============================================================
# APPLY FILTERS
# ============================================================

filtered_logs = logs


if selected_route != "All":

    filtered_logs = [
        log
        for log in filtered_logs
        if log.get("route") == selected_route
    ]


if selected_user != "All":

    filtered_logs = [
        log
        for log in filtered_logs
        if log.get("user_id") == selected_user
    ]


# ============================================================
# LOG TABLE
# ============================================================

st.subheader(
    "Interaction History"
)

st.dataframe(
    [
        {
            "Timestamp": log.get("timestamp"),
            "User": log.get("user_id"),
            "Session": log.get("session_id"),
            "Route": log.get("route"),
            "Query": log.get("query"),
            "Model": log.get("model"),
            "Tokens": log.get("total_tokens"),
            "Latency": log.get("latency_seconds")
        }
        for log in filtered_logs
    ],
    use_container_width=True,
    hide_index=True
)


# ============================================================
# STRUCTURED JSON
# ============================================================

st.subheader(
    "Structured JSON"
)

for index, log in enumerate(
    filtered_logs,
    start=1
):

    with st.expander(
        f"{index}. {log.get('query', '')}"
    ):

        st.json(log)