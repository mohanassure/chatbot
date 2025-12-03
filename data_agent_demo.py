import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
import sseclient
import streamlit as st
from models import (
    ChartEventData,
    DataAgentRunRequest,
    ErrorEventData,
    Message,
    MessageContentItem,
    StatusEventData,
    TableEventData,
    TextContentItem,
    TextDeltaEventData,
    ThinkingDeltaEventData,
    ThinkingEventData,
    ToolResultEventData,
    ToolUseEventData,
)

# ------------------------------
# Environment variables
# ------------------------------
PAT = os.getenv("CORTEX_AGENT_DEMO_PAT")
HOST = os.getenv("CORTEX_AGENT_DEMO_HOST")
DATABASE = os.getenv("CORTEX_AGENT_DEMO_DATABASE", "SNOWFLAKE_INTELLIGENCE")
SCHEMA = os.getenv("CORTEX_AGENT_DEMO_SCHEMA", "AGENTS")
AGENT = os.getenv("CORTEX_AGENT_DEMO_AGENT", "SALES_INTELLIGENCE_AGENT")

# ------------------------------
# Azure Function URL to fetch Qlik filters
# ------------------------------
AZURE_FUNCTION_URL = "https://qlik-filters-backend-dna9eke8e9gbewda.eastus-01.azurewebsites.net/api/FiltersFunction?code=UuAmdppxRSEKRAOQVAcW6zwz-DaV0iHUZTMEuqfkA5LPAzFujnzhfA=="

if "qlik_filters" not in st.session_state:
    st.session_state.qlik_filters = []

if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------------------
# Agent call
# ------------------------------
def agent_run(prompt_messages) -> requests.Response:
    request_body = DataAgentRunRequest(
        model="claude-4-sonnet",
        messages=prompt_messages,
    )
    resp = requests.post(
        url=f"https://{HOST}/api/v2/databases/{DATABASE}/schemas/{SCHEMA}/agents/{AGENT}:run",
        data=request_body.to_json(),
        headers={
            "Authorization": f"Bearer {PAT}",
            "Content-Type": "application/json",
        },
        stream=True,
        verify=False,
    )
    if resp.status_code < 400:
        return resp
    else:
        raise Exception(f"Failed request with status {resp.status_code}: {resp.text}")

# ------------------------------
# Stream response events
# ------------------------------
def stream_events(response: requests.Response):
    content = st.container()
    content_map = defaultdict(content.empty)
    buffers = defaultdict(str)
    spinner = st.spinner("Waiting for response...")
    spinner.__enter__()

    events = sseclient.SSEClient(response).events()
    for event in events:
        match event.event:
            case "response.status":
                spinner.__exit__(None, None, None)
                data = StatusEventData.from_json(event.data)
                spinner = st.spinner(data.message)
                spinner.__enter__()
            case "response.text.delta":
                data = TextDeltaEventData.from_json(event.data)
                buffers[data.content_index] += data.text
                content_map[data.content_index].write(buffers[data.content_index])
            case "response.thinking.delta":
                data = ThinkingDeltaEventData.from_json(event.data)
                buffers[data.content_index] += data.text
                content_map[data.content_index].expander("Thinking", expanded=True).write(
                    buffers[data.content_index]
                )
            case "response.thinking":
                data = ThinkingDeltaEventData.from_json(event.data)
                content_map[data.content_index].expander("Thinking").write(data.text)
            case "response.tool_use":
                data = ToolUseEventData.from_json(event.data)
                content_map[data.content_index].expander("Tool use").json(data)
            case "response.tool_result":
                data = ToolResultEventData.from_json(event.data)
                content_map[data.content_index].expander("Tool result").json(data)
            case "response.chart":
                data = ChartEventData.from_json(event.data)
                spec = json.loads(data.chart_spec)
                content_map[data.content_index].vega_lite_chart(spec, use_container_width=True)
            case "response.table":
                data = TableEventData.from_json(event.data)
                data_array = np.array(data.result_set.data)
                column_names = [
                    col.name
                    for col in data.result_set.result_set_meta_data.row_type
                ]
                content_map[data.content_index].dataframe(
                    pd.DataFrame(data_array, columns=column_names)
                )
            case "error":
                data = ErrorEventData.from_json(event.data)
                st.error(f"Error: {data.message} (code: {data.code})")
                st.session_state.messages.pop()
                return
            case "response":
                data = Message.from_json(event.data)
                st.session_state.messages.append(data)
    spinner.__exit__(None, None, None)

# ------------------------------
# Process user message (with live filters)
# ------------------------------
def process_new_message(user_prompt: str):

    # ----- Fetch latest filters from Azure Function -----
    try:
        resp = requests.post(AZURE_FUNCTION_URL, json={})
        raw_data = resp.json()
        raw_filters = raw_data.get("filters", [])
    except Exception:
        raw_filters = []

    # Normalize filters
    normalized_filters = []
    for f in raw_filters:
        if not isinstance(f, dict):
            continue
        field = f.get("field", "unknown")
        values = f.get("values") or f.get("selectedValues") or []

        if isinstance(values, str):
            values = [v.strip() for v in values.split(",") if v.strip()]
        elif not isinstance(values, list):
            values = [values]

        if values:
            normalized_filters.append({"field": field, "values": values})

    st.session_state.qlik_filters = normalized_filters

    # ----- Build final prompt with SQL-style filter -----
    full_prompt = user_prompt
    if normalized_filters:
        where_clauses = []
        for f in normalized_filters:
            values_sql = ", ".join([f"'{v}'" for v in f["values"]])
            where_clauses.append(f"{f['field']} IN ({values_sql})")
        # Append directly without extra text
        full_prompt = f"{user_prompt} {' AND '.join(where_clauses)}"

    # Send message to agent
    message = Message(
        role="user",
        content=[MessageContentItem(TextContentItem(type="text", text=full_prompt))],
    )
    st.session_state.messages.append(message)

    render_message(message)

    with st.chat_message("assistant"):
        with st.spinner("Sending request..."):
            response = agent_run(st.session_state.messages)
        st.markdown(
            f"```request_id: {response.headers.get('X-Snowflake-Request-Id')}```"
        )
        stream_events(response)

# ------------------------------
# Render previous messages
# ------------------------------
def render_message(msg: Message):
    with st.chat_message(msg.role):
        for content_item in msg.content:
            match content_item.actual_instance.type:
                case "text":
                    st.markdown(content_item.actual_instance.text)
                case "chart":
                    spec = json.loads(content_item.actual_instance.chart.chart_spec)
                    st.vega_lite_chart(spec, use_container_width=True)
                case "table":
                    data_array = (
                        np.array(content_item.actual_instance.table.result_set.data)
                    )
                    column_names = [
                        col.name
                        for col in content_item.actual_instance.table.result_set.result_set_meta_data.row_type
                    ]
                    st.dataframe(pd.DataFrame(data_array, columns=column_names))
                case _:
                    st.expander(content_item.actual_instance.type).json(
                        content_item.actual_instance.to_json()
                    )

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("Cortex Agent")

for msg in st.session_state.messages:
    render_message(msg)

if user_input := st.chat_input("What is your question?"):
    process_new_message(user_input)
