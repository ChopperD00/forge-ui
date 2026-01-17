"""
SOLUS FORGE - Team Interface
A Streamlit-based POC for the FORGE multi-agent system
"""

import streamlit as st
import requests
import json
from datetime import datetime
import os

# Page config
st.set_page_config(
    page_title="SOLUS FORGE",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #FF6B35, #F7C59F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .agent-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #FF6B35;
    }
    .status-active {
        color: #00ff88;
        font-weight: bold;
    }
    .status-pending {
        color: #ffaa00;
        font-weight: bold;
    }
    .task-box {
        background: #0e1117;
        border: 1px solid #262730;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .response-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #FF6B35;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'task_history' not in st.session_state:
    st.session_state.task_history = []
if 'current_response' not in st.session_state:
    st.session_state.current_response = None

# Sidebar - Agent Status
with st.sidebar:
    st.markdown("## 🔥 SOLUS FORGE")
    st.markdown("*Multi-Agent Orchestration*")
    st.markdown("---")

    st.markdown("### 🤖 Agent Status")

    agents = [
        {"name": "forge-chief", "model": "Opus", "status": "active", "icon": "🎖️", "api": "Claude"},
        {"name": "forge-planner", "model": "Haiku", "status": "active", "icon": "⚡", "api": "Claude"},
        {"name": "forge-reviewer", "model": "Sonnet", "status": "active", "icon": "🔍", "api": "Claude"},
        {"name": "forge-researcher", "model": "Sonar", "status": "active", "icon": "📋", "api": "Perplexity"},
        {"name": "forge-creative", "model": "Flash 2.0", "status": "active", "icon": "🎨", "api": "Gemini"},
    ]

    for agent in agents:
        status_class = "status-active" if agent["status"] == "active" else "status-pending"
        st.markdown(f"""
        <div class="agent-card">
            <strong>{agent['icon']} {agent['name']}</strong><br>
            <small>Model: {agent['model']} ({agent['api']})</small><br>
            <span class="{status_class}">● {agent['status'].upper()}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎙️ Voice Output")
    st.markdown("**Vivian** (Australian)")
    st.markdown("*ElevenLabs TTS*")

    st.markdown("---")
    st.markdown("### 💬 Slack")
    st.markdown("`#forge-agents`")
    st.markdown("<small>Channel: C0A95E9UFK5</small>", unsafe_allow_html=True)

# Main content
st.markdown('<h1 class="main-header">🔥 SOLUS FORGE</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888;'>Multi-Agent Orchestration System</p>", unsafe_allow_html=True)

# Configuration
col1, col2 = st.columns([3, 1])

with col2:
    st.markdown("### ⚙️ Settings")

    # n8n endpoint configuration
    n8n_endpoint = st.text_input(
        "n8n Webhook URL",
        value=os.getenv("N8N_WEBHOOK_URL", "http://localhost:5678/webhook/forge-task"),
        help="The webhook endpoint for your n8n FORGE workflow"
    )

    # Agent routing preference
    routing_mode = st.selectbox(
        "Routing Mode",
        ["Auto (AI Classifier)", "Manual Selection"],
        help="How tasks are routed to agents"
    )

    if routing_mode == "Manual Selection":
        selected_agent = st.selectbox(
            "Select Agent",
            ["forge-chief", "forge-planner", "forge-reviewer", "forge-researcher", "forge-creative"]
        )
    else:
        selected_agent = None

    # Voice output toggle
    voice_enabled = st.toggle("🎙️ Voice Output", value=True)

    # Slack notification toggle
    slack_notify = st.toggle("💬 Slack Notifications", value=True)

with col1:
    st.markdown("### 📝 Submit Task")

    # Task input
    task_input = st.text_area(
        "Enter your task",
        height=150,
        placeholder="Examples:\n• Research the latest AI agent frameworks for 2026\n• Create a mockup for a dashboard UI\n• Review this code for security vulnerabilities\n• Break down this feature into implementation steps"
    )

    # Task type hints
    task_type = st.radio(
        "Task Type (optional hint)",
        ["Auto-detect", "🔍 Research", "🎨 Creative/Design", "📝 Code Review", "⚡ Planning", "🎖️ Complex/Strategy"],
        horizontal=True
    )

    # Submit button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])

    with col_btn1:
        submit_btn = st.button("🚀 Submit Task", type="primary", use_container_width=True)

    with col_btn2:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)

    if clear_btn:
        st.session_state.current_response = None
        st.rerun()

    if submit_btn and task_input:
        with st.spinner("🔄 Processing task..."):
            # Prepare payload
            payload = {
                "task": task_input,
                "timestamp": datetime.now().isoformat(),
                "voice_enabled": voice_enabled,
                "slack_notify": slack_notify
            }

            if routing_mode == "Manual Selection" and selected_agent:
                payload["agent"] = selected_agent

            if task_type != "Auto-detect":
                type_map = {
                    "🔍 Research": "research",
                    "🎨 Creative/Design": "creative",
                    "📝 Code Review": "review",
                    "⚡ Planning": "planning",
                    "🎖️ Complex/Strategy": "chief"
                }
                payload["task_type"] = type_map.get(task_type, "auto")

            try:
                # Send to n8n webhook
                response = requests.post(
                    n8n_endpoint,
                    json=payload,
                    timeout=120
                )

                if response.status_code == 200:
                    result = response.json()
                    st.session_state.current_response = result
                    st.session_state.task_history.append({
                        "task": task_input,
                        "response": result,
                        "timestamp": datetime.now().isoformat()
                    })
                    st.success("✅ Task completed!")
                else:
                    st.error(f"❌ Error: {response.status_code} - {response.text}")

            except requests.exceptions.ConnectionError:
                st.warning("⚠️ Could not connect to n8n. Running in demo mode...")
                # Demo response for testing without n8n
                agent_guess = "forge-researcher" if "research" in task_input.lower() else \
                             "forge-creative" if any(x in task_input.lower() for x in ["design", "mockup", "ui"]) else \
                             "forge-reviewer" if any(x in task_input.lower() for x in ["review", "code", "security"]) else \
                             "forge-planner" if any(x in task_input.lower() for x in ["plan", "break", "steps"]) else \
                             "forge-chief"

                demo_response = {
                    "success": True,
                    "agent": agent_guess,
                    "response": f"""## Demo Response

**Agent**: {agent_guess}
**Task**: {task_input[:200]}{'...' if len(task_input) > 200 else ''}

---

This is a **demo response** because n8n is not connected yet.

In production, your task would be:
1. 📥 Received by the n8n webhook
2. 🤖 Classified by the Task Classifier (Sonnet)
3. ➡️ Routed to the appropriate specialist agent
4. 🎙️ Converted to voice via Vivian (ElevenLabs)
5. 💬 Notified in #forge-agents Slack channel

**To enable full functionality:**
1. Deploy n8n locally or to cloud
2. Import the FORGE workflow JSON
3. Update the webhook URL in settings

---
*SOLUS FORGE Demo Mode*""",
                    "voice_generated": voice_enabled,
                    "slack_notified": slack_notify,
                    "demo_mode": True
                }
                st.session_state.current_response = demo_response
                st.session_state.task_history.append({
                    "task": task_input,
                    "response": demo_response,
                    "timestamp": datetime.now().isoformat()
                })

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Display current response
if st.session_state.current_response:
    st.markdown("---")
    st.markdown("### 📤 Response")

    response = st.session_state.current_response

    # Demo mode banner
    if response.get("demo_mode"):
        st.info("🔧 **Demo Mode** - Connect n8n for full functionality")

    # Response metadata
    meta_col1, meta_col2, meta_col3 = st.columns(3)
    with meta_col1:
        agent_used = response.get("agent", "Unknown")
        st.metric("Agent Used", agent_used)
    with meta_col2:
        voice_status = "✅ Generated" if response.get("voice_generated") else "❌ Disabled"
        st.metric("Voice Output", voice_status)
    with meta_col3:
        slack_status = "✅ Sent" if response.get("slack_notified") else "❌ Disabled"
        st.metric("Slack Notification", slack_status)

    # Response content
    st.markdown('<div class="response-box">', unsafe_allow_html=True)
    st.markdown(response.get("response", "No response content"))
    st.markdown('</div>', unsafe_allow_html=True)

    # Audio player (if voice was generated)
    if response.get("audio_url"):
        st.audio(response["audio_url"])

# Task History
if st.session_state.task_history:
    st.markdown("---")
    with st.expander("📜 Task History", expanded=False):
        for i, item in enumerate(reversed(st.session_state.task_history[-10:])):
            st.markdown(f"""
            <div class="task-box">
                <strong>Task #{len(st.session_state.task_history) - i}</strong>
                <small>({item['timestamp'][:19]})</small><br>
                <code>{item['task'][:100]}{'...' if len(item['task']) > 100 else ''}</code>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <small>
        🔥 SOLUS FORGE v1.3.1 |
        Voice: Vivian (ElevenLabs) |
        Slack: #forge-agents
    </small>
</div>
""", unsafe_allow_html=True)
