# 🔥 SOLUS FORGE - Team Interface

A Streamlit-based dashboard for the SOLUS FORGE multi-agent system.

## Features

- 🤖 **Agent Status Panel** - View all 5 FORGE agents and their status
- 📝 **Task Submission** - Submit tasks with optional routing hints
- 🎙️ **Voice Toggle** - Enable/disable Vivian voice output (ElevenLabs)
- 💬 **Slack Integration** - Toggle notifications to `#forge-agents`
- 📜 **Task History** - View recent submissions
- 🎨 **Dark Theme** - FORGE-branded colors

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open: http://localhost:8501

## Deploy to Streamlit Cloud

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub
4. Select this repo and `app.py`
5. Deploy!

## Agent Team

| Agent | Model | API | Role |
|-------|-------|-----|------|
| 🎖️ forge-chief | Opus | Claude | Orchestration |
| ⚡ forge-planner | Haiku | Claude | Task decomposition |
| 🔍 forge-reviewer | Sonnet | Claude | Code review |
| 📋 forge-researcher | Sonar | Perplexity | Web research |
| 🎨 forge-creative | Flash 2.0 | Gemini | Visual design |

## Configuration

Set environment variable for n8n webhook:
```
N8N_WEBHOOK_URL=http://your-n8n-instance/webhook/forge-task
```

---

*SOLUS FORGE v1.3.1*
