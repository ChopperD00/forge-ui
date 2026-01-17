"""
SOLUS FORGE - Creative Command Center v1.5.1
"""
import streamlit as st
import os
import requests
import base64
from datetime import datetime

st.set_page_config(page_title="SOLUS FORGE", page_icon="🔥", layout="wide")

def load_styles():
    st.markdown("""<style>
    .main-header{font-size:2.5rem;font-weight:bold;background:linear-gradient(90deg,#FF6B35,#F7C59F);-webkit-background-clip:text;-webkit-text-fill-color:transparent;text-align:center}
    .tool-card{background:#1e1e2e;border-radius:12px;padding:1.2rem;margin:0.5rem 0;border:1px solid #333}
    .status-active{color:#00ff88;font-weight:bold}
    .status-pending{color:#ffaa00;font-weight:bold}
    .badge-mcp{background:#9333ea33;color:#a855f7;padding:0.25rem 0.75rem;border-radius:20px;font-size:0.8rem}
    .badge-api{background:#3b82f633;color:#60a5fa;padding:0.25rem 0.75rem;border-radius:20px;font-size:0.8rem}
    .badge-setup{background:#ffaa0033;color:#ffaa00;padding:0.25rem 0.75rem;border-radius:20px;font-size:0.8rem}
    </style>""", unsafe_allow_html=True)

def init_state():
    if 'task_history' not in st.session_state:
        st.session_state.task_history = []
    if 'api_keys' not in st.session_state:
        st.session_state.api_keys = {'krea':'','runway':'','luma':'','elevenlabs':'','suno':''}
    if 'generated_audio' not in st.session_state:
        st.session_state.generated_audio = None

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🔥 SOLUS FORGE")
        st.markdown("*Creative Command Center*")
        st.markdown("---")
        page = st.radio("Navigation", ["Command Center", "Adobe Suite", "Video Stack", "Audio / Soundwave"])
        st.markdown("---")
        st.markdown("### ✅ MCP Connected")
        st.markdown("- After Effects")
        st.markdown("- Adobe Express")
        st.markdown("- Figma")
        st.markdown("### ⚠️ Needs Setup")
        st.markdown("- Photoshop")
        st.markdown("- Illustrator")
        st.markdown("- InDesign")
        return page

def page_command():
    st.markdown('<h1 class="main-header">🔥 SOLUS FORGE v1.5.1</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#888'>Multi-Agent Creative Orchestration</p>", unsafe_allow_html=True)
    agents = [("forge-chief","Opus","Claude"),("forge-planner","Haiku","Claude"),("forge-reviewer","Sonnet","Claude"),("forge-researcher","Sonar","Perplexity"),("forge-creative","Flash","Gemini")]
    cols = st.columns(5)
    for i,(name,model,api) in enumerate(agents):
        with cols[i]:
            st.markdown(f'<div class="tool-card" style="text-align:center"><strong>{name}</strong><br><small>{model} ({api})</small><br><span class="status-active">● ACTIVE</span></div>', unsafe_allow_html=True)
    st.markdown("---")
    task = st.text_area("Enter task", height=100)
    if st.button("🚀 Submit", type="primary") and task:
        st.session_state.task_history.append({"task":task,"time":datetime.now().isoformat()})
        st.success("Task submitted to FORGE agents!")

def page_adobe():
    st.markdown('<h1 class="main-header">🎨 Adobe Suite</h1>', unsafe_allow_html=True)
    st.markdown("### ✅ Connected (MCP)")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown('<div class="tool-card"><strong>After Effects</strong> <span class="badge-mcp">MCP</span><br><span class="status-active">● Connected</span></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="tool-card"><strong>Adobe Express</strong> <span class="badge-mcp">MCP</span><br><span class="status-active">● Connected</span></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="tool-card"><strong>Figma</strong> <span class="badge-mcp">MCP</span><br><span class="status-active">● Connected</span></div>', unsafe_allow_html=True)
    st.markdown("### ⚠️ Needs Setup (adb-mcp)")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown('<div class="tool-card"><strong>Photoshop</strong> <span class="badge-setup">SETUP</span><br><code>uv run mcp install ps-mcp</code></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="tool-card"><strong>Illustrator</strong> <span class="badge-setup">SETUP</span><br><code>uv run mcp install ai-mcp</code></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="tool-card"><strong>InDesign</strong> <span class="badge-setup">SETUP</span><br><code>uv run mcp install id-mcp</code></div>', unsafe_allow_html=True)
    with st.expander("📖 Installation Guide for adb-mcp"):
        st.code("""# Clone the adb-mcp repository
git clone https://github.com/mikechambers/adb-mcp.git
cd adb-mcp

# Install dependencies
uv sync

# Install MCP servers for each app
uv run mcp install ps-mcp   # Photoshop
uv run mcp install ai-mcp   # Illustrator
uv run mcp install id-mcp   # InDesign""", language="bash")

def page_video():
    st.markdown('<h1 class="main-header">🎬 Video Stack</h1>', unsafe_allow_html=True)
    cols = st.columns(3)
    tools = [("Krea AI","Real-time generation","krea"),("Runway Gen-3","Text to video","runway"),("Luma Dream Machine","Cinematic AI","luma")]
    for i,(name,desc,key) in enumerate(tools):
        with cols[i]:
            st.markdown(f'<div class="tool-card"><strong>{name}</strong> <span class="badge-api">API</span><br><small>{desc}</small></div>', unsafe_allow_html=True)
    st.markdown("### 🔑 API Keys")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.text_input("Krea API Key", type="password", key="krea_key")
    with c2:
        st.text_input("Runway API Key", type="password", key="runway_key")
    with c3:
        st.text_input("Luma API Key", type="password", key="luma_key")

def generate_elevenlabs_speech(api_key, text, voice_id="luVEyhT3CocLZaLBps8v"):
    """Generate speech using ElevenLabs API"""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    try:
        response = requests.post(url, json=data, headers=headers, timeout=30)
        if response.status_code == 200:
            return response.content, None
        else:
            return None, f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return None, str(e)

def page_audio():
    st.markdown('<h1 class="main-header">🎵 Audio / Soundwave</h1>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)

    with c1:
        st.markdown("### 🗣️ ElevenLabs TTS")
        st.markdown('<div class="tool-card"><strong>Voice: Vivian</strong> <span class="badge-api">API</span><br><small>ID: luVEyhT3CocLZaLBps8v</small></div>', unsafe_allow_html=True)
        elevenlabs_key = st.text_input("ElevenLabs API Key", type="password", key="elevenlabs_key")
        tts_text = st.text_area("Text to speak", value="I am Solus Prime. Welcome to the Forge.", key="tts_text")

        if st.button("🔊 Generate Speech"):
            if not elevenlabs_key:
                st.error("Please enter your ElevenLabs API key")
            elif not tts_text:
                st.error("Please enter text to speak")
            else:
                with st.spinner("Generating speech..."):
                    audio_data, error = generate_elevenlabs_speech(elevenlabs_key, tts_text)
                    if error:
                        st.error(f"Failed to generate speech: {error}")
                    else:
                        st.session_state.generated_audio = audio_data
                        st.success("✅ Speech generated!")

        # Display audio player if we have generated audio
        if st.session_state.generated_audio:
            st.audio(st.session_state.generated_audio, format="audio/mpeg")
            # Download button
            st.download_button(
                label="⬇️ Download Audio",
                data=st.session_state.generated_audio,
                file_name="solus_forge_speech.mp3",
                mime="audio/mpeg"
            )

    with c2:
        st.markdown("### 🎼 Soundwave")
        st.markdown('<div class="tool-card"><strong>Music Generation</strong> <span class="badge-api">API</span><br><small>Suno V5 Integration</small></div>', unsafe_allow_html=True)
        st.text_input("Suno API Key", type="password", key="suno_key")
        st.text_area("Music prompt", key="music_prompt", placeholder="e.g., Upbeat electronic pop, female vocals, 120 BPM...")
        if st.button("🎵 Generate Music"):
            st.info("🚧 Suno API integration coming soon. Use Soundwave prompts directly in Suno for now.")

def main():
    load_styles()
    init_state()
    page = render_sidebar()
    if page == "Command Center":
        page_command()
    elif page == "Adobe Suite":
        page_adobe()
    elif page == "Video Stack":
        page_video()
    elif page == "Audio / Soundwave":
        page_audio()
    st.markdown("---")
    st.markdown("<p style='text-align:center;color:#666;font-size:0.8rem'>🔥 SOLUS FORGE v1.5.1 | Voice: Vivian (ElevenLabs) | Slack: #forge-agents</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
