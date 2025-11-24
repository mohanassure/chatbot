import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
import sseclient
import streamlit as st
from streamlit.components.v1 import html

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

# =========================
# Cortex Agent Config
# =========================
PAT = os.getenv("CORTEX_AGENT_DEMO_PAT")
HOST = os.getenv("CORTEX_AGENT_DEMO_HOST")
DATABASE = os.getenv("CORTEX_AGENT_DEMO_DATABASE", "SNOWFLAKE_INTELLIGENCE")
SCHEMA = os.getenv("CORTEX_AGENT_DEMO_SCHEMA", "AGENTS")
AGENT = os.getenv("CORTEX_AGENT_DEMO_AGENT", "SALES_INTELLIGENCE_AGENT")

# =========================
# Streamlit Session State
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "qlik_filters" not in st.session_state:
    st.session_state.qlik_filters = {}

# =========================
# JS Component to Receive Qlik Filters
# =========================
html("""
<script>
window.addEventListener("message", (event) => {
    if (event.data.type === "qlik_filters") {
        const filters = event.data.filters;
        // Update Streamlit session state via a hidden element
        const input = document.createElement('input');
        input.type = 'hidden';
        input.id = 'qlik_filters_input';
        input.value = JSON.stringify(filters);
        document.body.appendChild(input);
        // Send back to Streamlit
        window.parent.postMessage({type:"updateFilters", filters:filters}, "*");
    }
});
</script>
""", height=0)

# Optional: Poll hidden input to update session_state
filters_param = st.experimental_get_query_params().get("qlik_filters")
if filters_param:
    st.session_state.qlik_filters = json.loads(filters_param[0])

# =========================
# Cortex Agent Functions
# =========================
def agent_run() -> requests.Response:
    request_body = DataAgentRunRequest(
        model="claude-4-sonnet",
        messages=st.session_state.messages,
    )
    resp = requests.post(
        url=f"https://{HOST}/api/v2/databases/{DATABASE}/schemas/{SCHEMA}/agents/{AGENT}:run",
        data=request_body.to_json(),
        headers={
            "Authorization": f'Bearer {PAT}',
            "Content-Type": "application/json",
        },
        stream=True,
        verify=False,
    )
    if resp.status_code < 400:
        return resp
    else:
        raise Exception(f"Failed request with status {resp.status_code}: {resp.text}")


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
                content_map[data.content_index].expander("Thinking", expanded=True).write(buffers[data.content_index])
            case "response.thinking":
                data = ThinkingEventData.from_json(event.data)
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
                column_names = [col.name for col in data.result_set.result_set_meta_data.row_type]
                content_map[data.content_index].dataframe(pd.DataFrame(data_array, columns=column_names))
            case "error":
                data = ErrorEventData.from_json(event.data)
                st.error(f"Error: {data.message} (code: {data.code})")
                st.session_state.messages.pop()
                return
            case "response":
                data = Message.from_json(event.data)
                st.session_state.messages.append(data)
    spinner.__exit__(None, None, None)


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
                    data_array = np.array(content_item.actual_instance.table.result_set.data)
                    column_names = [col.name for col in content_item.actual_instance.table.result_set.result_set_meta_data.row_type]
                    st.dataframe(pd.DataFrame(data_array, columns=column_names))
                case _:
                    st.expander(content_item.actual_instance.type).json(content_item.actual_instance.to_json())


def process_new_message(prompt: str) -> None:
    """Process user message with optional Qlik filters."""
    filters_text = ""
    if st.session_state.get("qlik_filters"):
        filters_list = [f"{f['field']} = {f['value']}" for f in st.session_state.qlik_filters]
        if filters_list:
            filters_text = " Filters applied: " + ", ".join(filters_list)

    full_prompt = prompt + filters_text

    message = Message(
        role="user",
        content=[MessageContentItem(TextContentItem(type="text", text=full_prompt))],
    )
    render_message(message)
    st.session_state.messages.append(message)

    with st.chat_message("assistant"):
        with st.spinner("Sending request..."):
            response = agent_run()
        st.markdown(f"```request_id: {response.headers.get('X-Snowflake-Request-Id')}```")
        stream_events(response)

# =========================
# Streamlit UI
# =========================
st.title("Cortex Agent Chatbot")

for message in st.session_state.messages:
    render_message(message)

if user_input := st.chat_input("What is your question?"):
    process_new_message(prompt=user_input)
