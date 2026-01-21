"""
SOLUS FORGE v2.0 - Creative Command Center
Updated: 2026-01-21
- New intent-based lander flow
- Streamlined stack: CyberFilm SAGA + Stability AI integration
- Modular workspace architecture
- Consolidated video/audio/3D generation
"""
import streamlit as st
import json
from datetime import datetime
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="SOLUS FORGE 2.0",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Stack Configuration
STACK_CONFIG = {
    "pre_production": {
        "CyberFilm SAGA": {
            "status": "new",
            "type": "platform",
            "desc": "Scripts → Storyboards → Video",
            "url": "https://writeonsaga.com",
            "capabilities": ["screenplay", "storyboard", "character_sheets", "video_gen"],
            "models": ["Veo 3.1", "Luma Ray-3", "KlingAI 2.1", "FLUX 1.1"]
        },
        "Saga.so": {
            "status": "active",
            "type": "tool",
            "desc": "Project management & notes",
            "url": "https://saga.so"
        }
    },
    "generation": {
        "image": {
            "Stability SD3.5": {
                "status": "new",
                "type": "api",
                "desc": "Professional image generation",
                "url": "https://stability.ai",
                "models": ["SD 3.5 Large", "SDXL"]
            },
            "Krea AI": {
                "status": "active",
                "type": "api",
                "desc": "Real-time iteration",
                "url": "https://krea.ai"
            }
        },
        "video": {
            "CyberFilm": {
                "status": "new",
                "type": "platform",
                "desc": "Story-to-video pipeline",
                "models": ["Veo 3.1", "Luma Ray-3", "Runway Gen-4"]
            },
            "Runway Gen-3": {
                "status": "active",
                "type": "api",
                "desc": "Quick clip generation",
                "url": "https://runwayml.com"
            }
        },
        "audio": {
            "ElevenLabs": {
                "status": "active",
                "type": "api",
                "desc": "Voice synthesis (Vivian)",
                "url": "https://elevenlabs.io",
                "voice_id": "luVEyhT3CocLZaLBps8v"
            },
            "Stability Audio": {
                "status": "new",
                "type": "api",
                "desc": "SFX, soundscapes, music",
                "url": "https://stability.ai"
            },
            "Suno V5": {
                "status": "optional",
                "type": "api",
                "desc": "Full song generation",
                "url": "https://suno.ai"
            }
        },
        "3d": {
            "Stability SPAR3D": {
                "status": "new",
                "type": "api",
                "desc": "Image → 3D mesh (<1 sec)",
                "url": "https://stability.ai"
            }
        }
    },
    "post_production": {
        "After Effects": {"status": "mcp", "type": "mcp", "desc": "Motion graphics, VFX"},
        "Premiere Pro": {"status": "mcp", "type": "mcp", "desc": "Video editing"},
        "Photoshop": {"status": "setup", "type": "mcp", "desc": "Image editing"},
        "Illustrator": {"status": "setup", "type": "mcp", "desc": "Vector graphics"},
        "Figma": {"status": "mcp", "type": "mcp", "desc": "UI/UX design"}
    },
    "avatars": {
        "HeyGen": {
            "status": "active",
            "type": "platform",
            "desc": "Avatar creation, brand kit",
            "url": "https://heygen.com"
        }
    }
}

INTENT_MODULES = [
    {
        "id": "brand",
        "icon": "🎨",
        "title": "Brand Assets",
        "subtitle": "Emails, social, brand guides",
        "tools": ["Stability SD3.5", "Krea AI", "Figma", "Illustrator"],
        "color": "#FF6B35"
    },
    {
        "id": "video_story",
        "icon": "🎬",
        "title": "Video from Story",
        "subtitle": "Script → Storyboard → Edit",
        "tools": ["CyberFilm SAGA", "After Effects", "Premiere"],
        "color": "#9333EA"
    },
    {
        "id": "audio",
        "icon": "🎵",
        "title": "Audio Production",
        "subtitle": "Music, VO, SFX",
        "tools": ["ElevenLabs", "Stability Audio", "Suno V5"],
        "color": "#3B82F6"
    },
    {
        "id": "image_gen",
        "icon": "🖼️",
        "title": "Image Generation",
        "subtitle": "SD3.5, Krea, FLUX",
        "tools": ["Stability SD3.5", "Krea AI", "CyberFilm FLUX"],
        "color": "#10B981"
    },
    {
        "id": "quick_clips",
        "icon": "⚡",
        "title": "Quick Clips",
        "subtitle": "Instant video generation",
        "tools": ["Runway Gen-3", "Luma Dream Machine"],
        "color": "#F59E0B"
    },
    {
        "id": "avatar",
        "icon": "👤",
        "title": "Avatar & Presenter",
        "subtitle": "HeyGen clones, brand kit",
        "tools": ["HeyGen", "ElevenLabs"],
        "color": "#EC4899"
    }
]

# ============================================================================
# STYLES
# ============================================================================

def load_styles():
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FF6B35 0%, #F7C59F 50%, #FF6B35 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .intent-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .intent-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16162a 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid #333;
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: center;
    }
    
    .intent-card:hover {
        transform: translateY(-4px);
        border-color: #FF6B35;
        box-shadow: 0 8px 32px rgba(255, 107, 53, 0.2);
    }
    
    .intent-icon {
        font-size: 2.5rem;
        margin-bottom: 0.75rem;
    }
    
    .intent-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #fff;
        margin-bottom: 0.25rem;
    }
    
    .intent-subtitle {
        font-size: 0.85rem;
        color: #888;
    }
    
    .stack-section {
        background: #1a1a2e;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #333;
    }
    
    .stack-title {
        font-size: 1rem;
        font-weight: 600;
        color: #FF6B35;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .tool-pill {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.25rem;
        font-weight: 500;
    }
    
    .pill-new { background: #10B98133; color: #10B981; border: 1px solid #10B981; }
    .pill-active { background: #3B82F633; color: #60A5FA; border: 1px solid #3B82F6; }
    .pill-mcp { background: #9333EA33; color: #A855F7; border: 1px solid #9333EA; }
    .pill-setup { background: #F59E0B33; color: #FBBF24; border: 1px solid #F59E0B; }
    .pill-optional { background: #6B728033; color: #9CA3AF; border: 1px solid #6B7280; }
    
    .workspace-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem;
        background: #0d0d15;
        border-radius: 12px 12px 0 0;
        border: 1px solid #333;
        border-bottom: none;
    }
    
    .workspace-canvas {
        background: #0a0a0f;
        border: 1px solid #333;
        border-radius: 0 0 12px 12px;
        min-height: 400px;
        padding: 1rem;
    }
    
    .model-selector {
        background: #1a1a2e;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        border: 1px solid #333;
    }
    
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 0.5rem;
    }
    
    .dot-green { background: #10B981; }
    .dot-yellow { background: #F59E0B; }
    .dot-purple { background: #9333EA; }
    .dot-gray { background: #6B7280; }
    
    .quick-action-btn {
        background: linear-gradient(135deg, #FF6B35, #FF8F5A);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .quick-action-btn:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 16px rgba(255, 107, 53, 0.3);
    }
    
    .cyberfilm-highlight {
        background: linear-gradient(135deg, #1a1a2e, #2d1f3d);
        border: 2px solid #9333EA;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .stability-highlight {
        background: linear-gradient(135deg, #1a1a2e, #1f2d3d);
        border: 2px solid #3B82F6;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    </style>""", unsafe_allow_html=True)

# ============================================================================
# STATE MANAGEMENT
# ============================================================================

def init_state():
    defaults = {
        'current_view': 'lander',
        'selected_intent': None,
        'task_history': [],
        'api_keys': {},
        'mcp_status': check_mcp_status(),
        'workspace_model': 'claude-sonnet',
        'generated_content': {}
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def check_mcp_status():
    """Check which MCP servers are configured"""
    status = {}
    config_paths = [
        Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json",
        Path.home() / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json"
    ]
    
    for config_path in config_paths:
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                    servers = config.get('mcpServers', {})
                    status['after-effects'] = 'after-effects' in servers
                    status['adobe-express'] = 'adobe-express' in servers
                    status['figma'] = 'figma' in servers
                    status['photoshop'] = any('ps-mcp' in str(v) for v in servers.values())
                    status['illustrator'] = any('ai-mcp' in str(v) for v in servers.values())
                    status['premiere'] = any('premiere' in k.lower() for k in servers.keys())
            except:
                pass
    return status

# ============================================================================
# COMPONENTS
# ============================================================================

def render_intent_card(intent):
    """Render a single intent selection card"""
    return f'''
    <div class="intent-card" style="border-left: 3px solid {intent['color']}">
        <div class="intent-icon">{intent['icon']}</div>
        <div class="intent-title">{intent['title']}</div>
        <div class="intent-subtitle">{intent['subtitle']}</div>
    </div>
    '''

def render_tool_pill(name, status):
    """Render a tool status pill"""
    status_class = {
        'new': 'pill-new',
        'active': 'pill-active',
        'mcp': 'pill-mcp',
        'setup': 'pill-setup',
        'optional': 'pill-optional'
    }.get(status, 'pill-active')
    
    label = {'new': '✨ NEW', 'mcp': '🔌 MCP', 'setup': '⚠️ SETUP'}.get(status, '')
    
    return f'<span class="tool-pill {status_class}">{name} {label}</span>'

# ============================================================================
# PAGES
# ============================================================================

def page_lander():
    """Main landing page with intent selection"""
    st.markdown('<h1 class="main-header">🔥 SOLUS FORGE</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">What are you building today?</p>', unsafe_allow_html=True)
    
    # Intent Grid
    cols = st.columns(3)
    for i, intent in enumerate(INTENT_MODULES):
        with cols[i % 3]:
            if st.button(
                f"{intent['icon']} {intent['title']}\n{intent['subtitle']}",
                key=f"intent_{intent['id']}",
                use_container_width=True,
                type="secondary"
            ):
                st.session_state.selected_intent = intent['id']
                st.session_state.current_view = 'workspace'
                st.rerun()
    
    st.markdown("---")
    
    # New Additions Highlight
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('''
        <div class="cyberfilm-highlight">
            <h3>✨ NEW: CyberFilm SAGA</h3>
            <p style="color:#aaa;margin:0.5rem 0">All-in-one filmmaking platform</p>
            <ul style="color:#888;font-size:0.9rem">
                <li>AI Copilot (GPT-4o) for scripts</li>
                <li>Auto-storyboards (FLUX 1.1, Imagen 4)</li>
                <li>Video gen: Veo 3.1, Luma Ray-3, Runway Gen-4</li>
                <li>Character sheets & beat outlines</li>
            </ul>
            <p style="color:#9333EA;font-size:0.85rem;margin-top:0.75rem">$19.99/mo or Free tier</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown('''
        <div class="stability-highlight">
            <h3>✨ NEW: Stability AI Suite</h3>
            <p style="color:#aaa;margin:0.5rem 0">Multimodal generation platform</p>
            <ul style="color:#888;font-size:0.9rem">
                <li>SD 3.5 Large - pro image generation</li>
                <li>Stable Audio 2.5 - SFX & soundscapes</li>
                <li>SPAR3D - image to 3D in <1 second</li>
                <li>Stable Virtual Camera - 2D to navigable 3D</li>
            </ul>
            <p style="color:#3B82F6;font-size:0.85rem;margin-top:0.75rem">Free Community License (<$1M rev)</p>
        </div>
        ''', unsafe_allow_html=True)
    
    # Quick Stack Overview
    st.markdown("### 📦 Current Stack")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Pre-Production", "Generation", "Post-Production", "Avatars"])
    
    with tab1:
        pills = []
        for name, info in STACK_CONFIG["pre_production"].items():
            pills.append(render_tool_pill(name, info["status"]))
        st.markdown(" ".join(pills), unsafe_allow_html=True)
    
    with tab2:
        for category, tools in STACK_CONFIG["generation"].items():
            st.markdown(f"**{category.upper()}**")
            pills = [render_tool_pill(name, info["status"]) for name, info in tools.items()]
            st.markdown(" ".join(pills), unsafe_allow_html=True)
    
    with tab3:
        pills = [render_tool_pill(name, info["status"]) for name, info in STACK_CONFIG["post_production"].items()]
        st.markdown(" ".join(pills), unsafe_allow_html=True)
    
    with tab4:
        pills = [render_tool_pill(name, info["status"]) for name, info in STACK_CONFIG["avatars"].items()]
        st.markdown(" ".join(pills), unsafe_allow_html=True)

def page_workspace():
    """Modular workspace view"""
    intent = next((i for i in INTENT_MODULES if i['id'] == st.session_state.selected_intent), INTENT_MODULES[0])
    
    # Header
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if st.button("← Back"):
            st.session_state.current_view = 'lander'
            st.session_state.selected_intent = None
            st.rerun()
    with col2:
        st.markdown(f"<h2 style='text-align:center'>{intent['icon']} {intent['title']}</h2>", unsafe_allow_html=True)
    with col3:
        st.selectbox("Model", ["Claude Sonnet", "Claude Opus", "Claude Haiku", "GPT-4o"], key="model_select", label_visibility="collapsed")
    
    st.markdown("---")
    
    # Workspace Layout
    col_canvas, col_panel = st.columns([2, 1])
    
    with col_canvas:
        st.markdown("### 🎨 Canvas")
        
        # Intent-specific workspace
        if intent['id'] == 'video_story':
            render_video_story_workspace()
        elif intent['id'] == 'audio':
            render_audio_workspace()
        elif intent['id'] == 'image_gen':
            render_image_workspace()
        elif intent['id'] == 'brand':
            render_brand_workspace()
        elif intent['id'] == 'quick_clips':
            render_quickclips_workspace()
        elif intent['id'] == 'avatar':
            render_avatar_workspace()
        else:
            st.info("Select a workflow to begin")
    
    with col_panel:
        st.markdown("### 🧰 Tools")
        for tool in intent['tools']:
            status = get_tool_status(tool)
            dot_class = {'new': 'dot-green', 'active': 'dot-green', 'mcp': 'dot-purple', 'setup': 'dot-yellow'}.get(status, 'dot-gray')
            st.markdown(f'<div class="model-selector"><span class="status-dot {dot_class}"></span>{tool}</div>', unsafe_allow_html=True)
        
        st.markdown("### 📁 Assets")
        st.file_uploader("Drop files here", accept_multiple_files=True, label_visibility="collapsed")
        
        st.markdown("### 📝 Notes")
        st.text_area("Session notes", height=100, label_visibility="collapsed", placeholder="Add notes about this project...")

def get_tool_status(tool_name):
    """Get status for a tool by name"""
    for category in STACK_CONFIG.values():
        if isinstance(category, dict):
            if tool_name in category:
                return category[tool_name].get('status', 'active')
            for sub in category.values():
                if isinstance(sub, dict) and tool_name in sub:
                    return sub[tool_name].get('status', 'active')
    return 'active'

def render_video_story_workspace():
    """Video from Story workflow"""
    st.markdown('''
    <div class="cyberfilm-highlight">
        <strong>🎬 CyberFilm SAGA Pipeline</strong>
        <p style="color:#888;font-size:0.9rem">Script → Character Sheets → Storyboard → Video</p>
    </div>
    ''', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Script", "👥 Characters", "🖼️ Storyboard", "🎥 Generate"])
    
    with tab1:
        st.text_area("Script / Treatment", height=200, placeholder="Write your story here... CyberFilm's AI Copilot will help with formatting, coverage, and suggestions.")
        col1, col2 = st.columns(2)
        with col1:
            st.button("✨ AI Coverage", use_container_width=True)
        with col2:
            st.button("📖 Format Screenplay", use_container_width=True)
    
    with tab2:
        st.text_input("Character Name")
        st.text_area("Character Description", height=100)
        st.button("🎨 Generate Character Sheet", use_container_width=True)
    
    with tab3:
        st.slider("Number of panels", 4, 24, 8)
        st.selectbox("Style", ["Cinematic", "Comic", "Anime", "Realistic"])
        st.button("🖼️ Generate Storyboard (FLUX 1.1)", use_container_width=True, type="primary")
    
    with tab4:
        st.selectbox("Video Model", ["Veo 3.1 (Best Quality)", "Luma Ray-3 (Fast)", "Runway Gen-4", "KlingAI 2.1"])
        st.slider("Duration (seconds)", 5, 60, 15)
        st.button("🎬 Generate Video", use_container_width=True, type="primary")

def render_audio_workspace():
    """Audio production workflow"""
    tab1, tab2, tab3 = st.tabs(["🗣️ Voice (ElevenLabs)", "🎵 Music (Suno)", "🔊 SFX (Stability)"])
    
    with tab1:
        st.markdown("**Voice: Vivian** (Australian, female)")
        text = st.text_area("Text to speak", value="Welcome to SOLUS FORGE.", height=100)
        api_key = st.text_input("ElevenLabs API Key", type="password")
        if st.button("🔊 Generate Speech", type="primary"):
            if api_key and text:
                st.info("Generating speech... (API integration ready)")
    
    with tab2:
        st.text_area("Music prompt", placeholder="Upbeat electronic, female vocals, 120 BPM...", height=100)
        st.selectbox("Genre", ["Electronic", "Pop", "Cinematic", "Lo-Fi", "Rock"])
        st.button("🎵 Generate Music (Suno V5)", use_container_width=True)
    
    with tab3:
        st.markdown('''
        <div class="stability-highlight">
            <strong>Stability Audio 2.5</strong>
            <p style="color:#888;font-size:0.85rem">Professional SFX, soundscapes, and audio inpainting</p>
        </div>
        ''', unsafe_allow_html=True)
        st.text_area("SFX Description", placeholder="Thunder rumbling, rain on window, distant traffic...", height=80)
        st.slider("Duration", 1, 60, 10, format="%d sec")
        st.button("🔊 Generate SFX", use_container_width=True, type="primary")

def render_image_workspace():
    """Image generation workflow"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.text_area("Prompt", height=150, placeholder="Describe the image you want to create...")
        st.text_area("Negative prompt", height=50, placeholder="What to avoid...")
    
    with col2:
        st.selectbox("Model", ["Stability SD 3.5 Large", "SDXL", "Krea Real-time", "FLUX 1.1 (via CyberFilm)"])
        st.selectbox("Aspect Ratio", ["1:1", "16:9", "9:16", "4:3", "3:2"])
        st.slider("Steps", 20, 50, 30)
        st.slider("CFG Scale", 1.0, 15.0, 7.5)
    
    if st.button("🖼️ Generate Image", type="primary", use_container_width=True):
        st.info("Image generation ready - add API key in settings")

def render_brand_workspace():
    """Brand assets workflow"""
    st.markdown("### Brand Asset Generator")
    
    asset_type = st.selectbox("Asset Type", ["Social Post", "Email Header", "Logo Variations", "Brand Guide Page"])
    
    if asset_type == "Social Post":
        col1, col2 = st.columns(2)
        with col1:
            st.selectbox("Platform", ["Instagram", "LinkedIn", "Twitter/X", "Facebook"])
            st.selectbox("Format", ["Square (1:1)", "Story (9:16)", "Landscape (16:9)"])
        with col2:
            st.text_input("Headline")
            st.text_area("Body copy", height=80)
    
    st.button("🎨 Generate Asset", type="primary", use_container_width=True)

def render_quickclips_workspace():
    """Quick video clip generation"""
    st.markdown('''
    <div style="background:#1a1a2e;padding:1rem;border-radius:8px;margin-bottom:1rem">
        <strong>⚡ Instant Video Generation</strong>
        <p style="color:#888;font-size:0.85rem;margin:0">Direct text-to-video without pre-production</p>
    </div>
    ''', unsafe_allow_html=True)
    
    st.text_area("Video prompt", height=100, placeholder="A golden retriever running through autumn leaves in slow motion...")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.selectbox("Model", ["Runway Gen-3 Alpha", "Luma Dream Machine", "Veo 3.1"])
    with col2:
        st.selectbox("Duration", ["4 sec", "8 sec", "16 sec"])
    with col3:
        st.selectbox("Aspect", ["16:9", "9:16", "1:1"])
    
    st.button("⚡ Generate Clip", type="primary", use_container_width=True)

def render_avatar_workspace():
    """Avatar creation workflow"""
    st.markdown('''
    <div style="background:#1a1a2e;padding:1rem;border-radius:8px;margin-bottom:1rem">
        <strong>👤 HeyGen Avatar Studio</strong>
        <p style="color:#888;font-size:0.85rem;margin:0">Create AI presenters and brand avatars</p>
    </div>
    ''', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Create Avatar", "Generate Video"])
    
    with tab1:
        st.file_uploader("Upload reference photo", type=["jpg", "png"])
        st.text_input("Avatar name")
        st.selectbox("Voice", ["Clone from audio", "Vivian (ElevenLabs)", "HeyGen Voice"])
    
    with tab2:
        st.selectbox("Select Avatar", ["Custom Avatar 1", "Stock - Professional Female", "Stock - Professional Male"])
        st.text_area("Script", height=150)
        st.button("🎥 Generate Avatar Video", type="primary", use_container_width=True)

def page_stack():
    """Full stack overview"""
    st.markdown('<h1 class="main-header">📦 Stack Overview</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">SOLUS FORGE v2.0 - Streamlined Creative Stack</p>', unsafe_allow_html=True)
    
    # Changes Summary
    st.markdown("### 🔄 What's Changed in v2.0")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### ➕ Added")
        st.markdown("""
        - **CyberFilm SAGA** - Script-to-video pipeline
        - **Stability SD 3.5** - Pro image generation
        - **Stability Audio 2.5** - SFX & soundscapes  
        - **Stability SPAR3D** - Instant 3D from images
        """)
    
    with col2:
        st.markdown("#### ➖ Consolidated")
        st.markdown("""
        - Luma Dream Machine → *Via CyberFilm (Luma Ray-3)*
        - Multiple storyboard tools → *CyberFilm native*
        - Some Suno use cases → *Stability Audio (SFX/ambient)*
        """)
    
    st.markdown("---")
    
    # Full Stack Display
    for section, tools in STACK_CONFIG.items():
        st.markdown(f"### {section.replace('_', ' ').title()}")
        
        if isinstance(tools, dict):
            # Check if nested (like generation)
            first_val = next(iter(tools.values()))
            if isinstance(first_val, dict) and 'status' not in first_val:
                # Nested categories
                for category, category_tools in tools.items():
                    st.markdown(f"**{category.upper()}**")
                    cols = st.columns(len(category_tools))
                    for i, (name, info) in enumerate(category_tools.items()):
                        with cols[i]:
                            status_color = {'new': '#10B981', 'active': '#3B82F6', 'mcp': '#9333EA', 'setup': '#F59E0B', 'optional': '#6B7280'}.get(info['status'], '#666')
                            st.markdown(f'''
                            <div style="background:#1a1a2e;padding:1rem;border-radius:8px;border-left:3px solid {status_color}">
                                <strong>{name}</strong><br>
                                <small style="color:#888">{info['desc']}</small><br>
                                <span style="color:{status_color};font-size:0.8rem">● {info['status'].upper()}</span>
                            </div>
                            ''', unsafe_allow_html=True)
            else:
                # Flat tools
                cols = st.columns(min(len(tools), 4))
                for i, (name, info) in enumerate(tools.items()):
                    with cols[i % 4]:
                        status_color = {'new': '#10B981', 'active': '#3B82F6', 'mcp': '#9333EA', 'setup': '#F59E0B', 'optional': '#6B7280'}.get(info['status'], '#666')
                        st.markdown(f'''
                        <div style="background:#1a1a2e;padding:1rem;border-radius:8px;border-left:3px solid {status_color}">
                            <strong>{name}</strong><br>
                            <small style="color:#888">{info['desc']}</small><br>
                            <span style="color:{status_color};font-size:0.8rem">● {info['status'].upper()}</span>
                        </div>
                        ''', unsafe_allow_html=True)
        
        st.markdown("")

def page_settings():
    """Settings and API keys"""
    st.markdown('<h1 class="main-header">⚙️ Settings</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🔑 API Keys", "🔌 MCP Status", "📖 Setup Guide"])
    
    with tab1:
        st.markdown("### Generation APIs")
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Stability AI", type="password", key="api_stability")
            st.text_input("Runway", type="password", key="api_runway")
            st.text_input("Krea AI", type="password", key="api_krea")
        with col2:
            st.text_input("ElevenLabs", type="password", key="api_elevenlabs")
            st.text_input("Suno", type="password", key="api_suno")
            st.text_input("HeyGen", type="password", key="api_heygen")
        
        st.markdown("### Platform Accounts")
        st.text_input("CyberFilm SAGA (email)", key="cyberfilm_email")
        
        if st.button("💾 Save API Keys"):
            st.success("API keys saved to session")
    
    with tab2:
        status = st.session_state.get('mcp_status', {})
        
        for app, connected in [
            ("After Effects", status.get('after-effects', False)),
            ("Adobe Express", status.get('adobe-express', False)),
            ("Figma", status.get('figma', False)),
            ("Photoshop", status.get('photoshop', False)),
            ("Illustrator", status.get('illustrator', False)),
            ("Premiere Pro", status.get('premiere', False))
        ]:
            icon = "✅" if connected else "⚠️"
            st.markdown(f"{icon} **{app}**: {'Connected' if connected else 'Not configured'}")
        
        if st.button("🔄 Refresh MCP Status"):
            st.session_state.mcp_status = check_mcp_status()
            st.rerun()
    
    with tab3:
        st.markdown("""
        ### Quick Setup Guide
        
        **1. CyberFilm SAGA**
        - Sign up at [writeonsaga.com](https://writeonsaga.com)
        - Free tier available (limited features)
        - Full access: $19.99/month
        
        **2. Stability AI**
        - Get API key at [stability.ai](https://stability.ai)
        - Free Community License for <$1M revenue
        - Includes SD 3.5, Audio 2.5, SPAR3D
        
        **3. Adobe MCP Setup**
        ```bash
        cd ~/Desktop/"Solus Forge 1.2"/adobe-mcp
        ./install-adobe-mcp.sh
        ```
        
        **4. Restart Claude Desktop** after MCP installation
        """)

# ============================================================================
# MAIN
# ============================================================================

def main():
    load_styles()
    init_state()
    
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("## 🔥 FORGE v2.0")
        st.markdown("---")
        
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.current_view = 'lander'
            st.session_state.selected_intent = None
            st.rerun()
        
        if st.button("📦 Stack", use_container_width=True):
            st.session_state.current_view = 'stack'
            st.rerun()
        
        if st.button("⚙️ Settings", use_container_width=True):
            st.session_state.current_view = 'settings'
            st.rerun()
        
        st.markdown("---")
        st.markdown("### Quick Launch")
        for intent in INTENT_MODULES[:3]:
            if st.button(f"{intent['icon']} {intent['title']}", key=f"quick_{intent['id']}", use_container_width=True):
                st.session_state.selected_intent = intent['id']
                st.session_state.current_view = 'workspace'
                st.rerun()
    
    # Route to current view
    view = st.session_state.current_view
    
    if view == 'lander':
        page_lander()
    elif view == 'workspace':
        page_workspace()
    elif view == 'stack':
        page_stack()
    elif view == 'settings':
        page_settings()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <p style='text-align:center;color:#555;font-size:0.8rem'>
        🔥 SOLUS FORGE v2.0 | 
        <a href="https://writeonsaga.com" target="_blank">CyberFilm</a> | 
        <a href="https://stability.ai" target="_blank">Stability AI</a> |
        Voice: Vivian (ElevenLabs)
    </p>
    """, unsafe_allow_html=True