"""
SOLUS FORGE v2.2 - Creative Command Center
Updated: 2026-01-21
- Shot List Ingest tab (paste, upload, or link to spreadsheets/docs)
- Visual Style Training tab (LoRA training for animation/style/character)
- New intent-based lander flow
- Streamlined stack: Direct APIs (no CyberFilm wrapper)
- Luma Dream Machine for iteration + Runway for polish
- Stability AI suite (SD3.5 + Audio 2.5 + SPAR3D)
- Modular workspace architecture
- Hybrid orchestration (ThreadPool + Claude Task tool)
"""
import streamlit as st
import json
import os
import uuid
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
from abc import ABC, abstractmethod

# ============================================================================
# ORCHESTRATION ENGINE
# ============================================================================

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class PipelineStage(Enum):
    PRE_PRODUCTION = "pre_production"
    GENERATION = "generation"
    POST_PRODUCTION = "post_production"
    EXPORT = "export"

@dataclass
class Task:
    """Individual task in the orchestration pipeline"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    stage: PipelineStage = PipelineStage.GENERATION
    tool: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    progress: float = 0.0
    dependencies: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    def duration(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None

@dataclass
class Pipeline:
    """Collection of tasks organized by stage"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = "Untitled Pipeline"
    tasks: List[Task] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)

    def add_task(self, task: Task) -> None:
        self.tasks.append(task)

    def get_tasks_by_stage(self, stage: PipelineStage) -> List[Task]:
        return [t for t in self.tasks if t.stage == stage]

    def get_ready_tasks(self) -> List[Task]:
        """Get tasks whose dependencies are all completed"""
        completed_ids = {t.id for t in self.tasks if t.status == TaskStatus.COMPLETED}
        return [
            t for t in self.tasks
            if t.status == TaskStatus.PENDING
            and all(dep in completed_ids for dep in t.dependencies)
        ]

    def progress(self) -> float:
        if not self.tasks:
            return 0.0
        completed = sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)
        return completed / len(self.tasks)

class BaseOrchestrator(ABC):
    """Abstract base class for orchestration backends"""

    @abstractmethod
    def execute_task(self, task: Task) -> Task:
        """Execute a single task"""
        pass

    @abstractmethod
    def execute_pipeline(self, pipeline: Pipeline, on_progress: Optional[Callable] = None) -> Pipeline:
        """Execute a full pipeline"""
        pass

class ThreadPoolOrchestrator(BaseOrchestrator):
    """
    Parallel task execution using ThreadPoolExecutor.
    Best for: Streamlit Cloud deployment, I/O-bound API calls.
    """

    def __init__(self, max_workers: int = 5):
        self.max_workers = max_workers
        self.executors: Dict[str, Callable] = {}
        self._register_default_executors()

    def _register_default_executors(self):
        """Register tool-specific execution functions"""
        self.executors = {
            # Pre-production
            "claude": self._exec_claude,
            "krea": self._exec_krea,
            # Generation - Image
            "stability_sd35": self._exec_stability_image,
            # Generation - Video
            "luma": self._exec_luma,
            "runway": self._exec_runway,
            # Generation - Audio
            "elevenlabs": self._exec_elevenlabs,
            "stability_audio": self._exec_stability_audio,
            "suno": self._exec_suno,
            # Generation - 3D
            "spar3d": self._exec_spar3d,
        }

    def register_executor(self, tool: str, executor: Callable):
        """Register a custom executor for a tool"""
        self.executors[tool] = executor

    def execute_task(self, task: Task) -> Task:
        """Execute a single task using the appropriate executor"""
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()

        try:
            executor = self.executors.get(task.tool)
            if executor:
                task.result = executor(task.params)
                task.status = TaskStatus.COMPLETED
                task.progress = 1.0
            else:
                raise ValueError(f"No executor registered for tool: {task.tool}")
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
        finally:
            task.completed_at = time.time()

        return task

    def execute_pipeline(self, pipeline: Pipeline, on_progress: Optional[Callable] = None) -> Pipeline:
        """
        Execute pipeline with parallel fan-out for independent tasks.
        Tasks within the same stage with no dependencies run in parallel.
        """
        pipeline.status = TaskStatus.RUNNING

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while True:
                ready_tasks = pipeline.get_ready_tasks()
                if not ready_tasks:
                    # Check if all tasks are done or if we're stuck
                    pending = [t for t in pipeline.tasks if t.status == TaskStatus.PENDING]
                    if not pending:
                        break
                    # If there are pending tasks but none are ready, we have a dependency issue
                    failed = [t for t in pipeline.tasks if t.status == TaskStatus.FAILED]
                    if failed:
                        break
                    time.sleep(0.1)
                    continue

                # Submit all ready tasks in parallel
                futures = {executor.submit(self.execute_task, task): task for task in ready_tasks}

                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        task.status = TaskStatus.FAILED
                        task.error = str(e)

                    if on_progress:
                        on_progress(pipeline.progress(), task)

        # Determine final pipeline status
        failed_tasks = [t for t in pipeline.tasks if t.status == TaskStatus.FAILED]
        if failed_tasks:
            pipeline.status = TaskStatus.FAILED
        else:
            pipeline.status = TaskStatus.COMPLETED

        return pipeline

    # === Tool Executors (Stubs - Replace with actual API calls) ===

    def _exec_claude(self, params: Dict) -> Dict:
        """Execute Claude LLM task"""
        # TODO: Integrate with Anthropic API
        return {"type": "claude", "content": f"Generated content for: {params.get('prompt', '')[:50]}..."}

    def _exec_krea(self, params: Dict) -> Dict:
        """Execute Krea AI real-time generation"""
        # TODO: Integrate with Krea API
        return {"type": "krea", "image_url": "https://placeholder.krea.ai/image.png"}

    def _exec_stability_image(self, params: Dict) -> Dict:
        """Execute Stability SD3.5 image generation"""
        # TODO: Integrate with Stability API
        return {"type": "stability_image", "image_url": "https://placeholder.stability.ai/image.png"}

    def _exec_luma(self, params: Dict) -> Dict:
        """Execute Luma Dream Machine video generation"""
        # TODO: Integrate with Luma API
        return {"type": "luma", "video_url": "https://placeholder.luma.ai/video.mp4"}

    def _exec_runway(self, params: Dict) -> Dict:
        """Execute Runway Gen-3 video generation"""
        # TODO: Integrate with Runway API
        return {"type": "runway", "video_url": "https://placeholder.runway.ai/video.mp4"}

    def _exec_elevenlabs(self, params: Dict) -> Dict:
        """Execute ElevenLabs voice synthesis"""
        # TODO: Integrate with ElevenLabs API
        return {"type": "elevenlabs", "audio_url": "https://placeholder.elevenlabs.io/audio.mp3"}

    def _exec_stability_audio(self, params: Dict) -> Dict:
        """Execute Stability Audio 2.5 SFX generation"""
        # TODO: Integrate with Stability Audio API
        return {"type": "stability_audio", "audio_url": "https://placeholder.stability.ai/audio.mp3"}

    def _exec_suno(self, params: Dict) -> Dict:
        """Execute Suno V5 music generation"""
        # TODO: Integrate with Suno API
        return {"type": "suno", "audio_url": "https://placeholder.suno.ai/music.mp3"}

    def _exec_spar3d(self, params: Dict) -> Dict:
        """Execute Stability SPAR3D image-to-3D"""
        # TODO: Integrate with SPAR3D API
        return {"type": "spar3d", "model_url": "https://placeholder.stability.ai/model.glb"}

class ClaudeTaskOrchestrator(BaseOrchestrator):
    """
    Orchestration using Claude Code/Cowork Task tool for sub-agents.
    Best for: Running inside Claude desktop app with full agent capabilities.
    """

    def __init__(self):
        self.is_available = self._check_claude_environment()

    def _check_claude_environment(self) -> bool:
        """Check if we're running in Claude Code/Cowork environment"""
        # Check for Claude Code indicators
        indicators = [
            os.environ.get("CLAUDE_CODE"),
            os.environ.get("ANTHROPIC_API_KEY"),
            Path.home().joinpath(".claude").exists(),
        ]
        return any(indicators)

    def execute_task(self, task: Task) -> Task:
        """
        Execute task using Claude sub-agent.
        In actual implementation, this would use the Task tool to spawn a sub-agent.
        """
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()

        try:
            # Build sub-agent prompt based on task
            agent_prompt = self._build_agent_prompt(task)

            # In real implementation:
            # result = Task(prompt=agent_prompt, subagent_type="general-purpose")
            # For now, simulate the result
            task.result = {
                "type": "claude_agent",
                "task": task.name,
                "prompt": agent_prompt[:100] + "...",
                "note": "Sub-agent execution requires Claude Code environment"
            }
            task.status = TaskStatus.COMPLETED
            task.progress = 1.0
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
        finally:
            task.completed_at = time.time()

        return task

    def _build_agent_prompt(self, task: Task) -> str:
        """Build prompt for sub-agent based on task type"""
        tool_prompts = {
            "claude": f"Creative writing task: {task.params.get('prompt', '')}",
            "luma": f"Generate video using Luma Dream Machine API: {task.params}",
            "runway": f"Polish video using Runway Gen-3 API: {task.params}",
            "stability_sd35": f"Generate image using Stability SD3.5: {task.params}",
            "elevenlabs": f"Generate voice audio using ElevenLabs: {task.params}",
        }
        return tool_prompts.get(task.tool, f"Execute {task.tool} with params: {task.params}")

    def execute_pipeline(self, pipeline: Pipeline, on_progress: Optional[Callable] = None) -> Pipeline:
        """
        Execute pipeline using Claude sub-agents.
        Can spawn multiple sub-agents in parallel for independent tasks.
        """
        pipeline.status = TaskStatus.RUNNING

        # Group tasks by stage for organized execution
        for stage in PipelineStage:
            stage_tasks = pipeline.get_tasks_by_stage(stage)
            if not stage_tasks:
                continue

            # Execute stage tasks (in real impl, would spawn parallel sub-agents)
            for task in stage_tasks:
                if task.status != TaskStatus.PENDING:
                    continue

                # Check dependencies
                dep_tasks = [t for t in pipeline.tasks if t.id in task.dependencies]
                if any(t.status != TaskStatus.COMPLETED for t in dep_tasks):
                    continue

                self.execute_task(task)

                if on_progress:
                    on_progress(pipeline.progress(), task)

        # Determine final status
        failed = [t for t in pipeline.tasks if t.status == TaskStatus.FAILED]
        pipeline.status = TaskStatus.FAILED if failed else TaskStatus.COMPLETED

        return pipeline

def get_orchestrator() -> BaseOrchestrator:
    """
    Factory function to get the appropriate orchestrator based on environment.
    Returns ClaudeTaskOrchestrator if in Claude Code, otherwise ThreadPoolOrchestrator.
    """
    claude_orch = ClaudeTaskOrchestrator()
    if claude_orch.is_available:
        return claude_orch
    return ThreadPoolOrchestrator()

def create_video_pipeline(script: str, style: str = "cinematic", duration: int = 15) -> Pipeline:
    """
    Factory function to create a standard video production pipeline.
    Demonstrates fan-out pattern: multiple generation tasks run in parallel.
    """
    pipeline = Pipeline(name=f"Video: {script[:30]}...")

    # Stage 1: Pre-production (sequential)
    script_task = Task(
        name="Develop Script",
        stage=PipelineStage.PRE_PRODUCTION,
        tool="claude",
        params={"prompt": script, "task": "expand_screenplay"}
    )
    pipeline.add_task(script_task)

    storyboard_task = Task(
        name="Generate Storyboard",
        stage=PipelineStage.PRE_PRODUCTION,
        tool="krea",
        params={"prompt": script, "style": style, "panels": 8},
        dependencies=[script_task.id]
    )
    pipeline.add_task(storyboard_task)

    # Stage 2: Generation (parallel fan-out)
    # These tasks can run simultaneously
    video_iteration = Task(
        name="Video Draft (Luma)",
        stage=PipelineStage.GENERATION,
        tool="luma",
        params={"prompt": script, "duration": duration},
        dependencies=[storyboard_task.id]
    )
    pipeline.add_task(video_iteration)

    voice_task = Task(
        name="Voice Narration",
        stage=PipelineStage.GENERATION,
        tool="elevenlabs",
        params={"text": script, "voice": "vivian"},
        dependencies=[script_task.id]  # Can start after script, parallel with video
    )
    pipeline.add_task(voice_task)

    sfx_task = Task(
        name="Sound Effects",
        stage=PipelineStage.GENERATION,
        tool="stability_audio",
        params={"prompt": f"SFX for: {script[:50]}", "duration": duration},
        dependencies=[script_task.id]  # Can start after script, parallel with video
    )
    pipeline.add_task(sfx_task)

    # Stage 3: Polish (after iteration)
    video_polish = Task(
        name="Video Polish (Runway)",
        stage=PipelineStage.GENERATION,
        tool="runway",
        params={"input_video": "luma_output", "enhance": True},
        dependencies=[video_iteration.id]
    )
    pipeline.add_task(video_polish)

    return pipeline

def create_image_pipeline(prompt: str, variations: int = 3) -> Pipeline:
    """
    Factory function to create an image generation pipeline with parallel variations.
    """
    pipeline = Pipeline(name=f"Image: {prompt[:30]}...")

    # Generate multiple variations in parallel
    for i in range(variations):
        task = Task(
            name=f"Variation {i+1}",
            stage=PipelineStage.GENERATION,
            tool="stability_sd35",
            params={"prompt": prompt, "seed": i * 1000, "variation": i}
        )
        pipeline.add_task(task)

    return pipeline

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="SOLUS FORGE 2.0",
    page_icon=":fire:",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Stack Configuration
STACK_CONFIG = {
    "pre_production": {
        "Claude Opus/Sonnet": {
            "status": "active",
            "type": "llm",
            "desc": "Scripts, treatments, character dev",
            "capabilities": ["screenplay", "creative_writing", "analysis"]
        },
        "Krea AI": {
            "status": "active",
            "type": "api",
            "desc": "Real-time storyboard iteration",
            "url": "https://krea.ai"
        },
        "Stability SD3.5": {
            "status": "active",
            "type": "api",
            "desc": "Final storyboard frames, char sheets",
            "url": "https://stability.ai"
        }
    },
    "generation": {
        "image": {
            "Stability SD3.5": {
                "status": "active",
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
            "Luma Dream Machine": {
                "status": "active",
                "type": "api",
                "desc": "Fast iteration, creative exploration",
                "url": "https://lumalabs.ai"
            },
            "Runway Gen-3": {
                "status": "active",
                "type": "api",
                "desc": "Polish, final renders",
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
            "Stability Audio 2.5": {
                "status": "active",
                "type": "api",
                "desc": "SFX, soundscapes, ambience",
                "url": "https://stability.ai"
            },
            "Suno V5": {
                "status": "optional",
                "type": "api",
                "desc": "Full song generation (optional)",
                "url": "https://suno.ai"
            }
        },
        "3d": {
            "Stability SPAR3D": {
                "status": "active",
                "type": "api",
                "desc": "Image -> 3D mesh (<1 sec)",
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
        "id": "email_template",
        "icon": "[EMAIL]",
        "title": "Email Template Editor",
        "subtitle": "Update copy & photos from annotated doc",
        "tools": ["Dropbox", "Google Docs", "Figma", "Stability SD3.5"],
        "input_type": "file",
        "placeholder": "Paste Dropbox or Google Doc link with annotations...",
        "color": "#FF6B35"
    },
    {
        "id": "video_localizer",
        "icon": "[VIDEO]",
        "title": "Video Localizer",
        "subtitle": "Append localized end cards to assets",
        "tools": ["After Effects", "Runway Gen-3", "ElevenLabs", "Luma"],
        "input_type": "batch",
        "placeholder": "Describe localization needs (regions, languages, end card specs)...",
        "color": "#9333EA"
    },
    {
        "id": "music_beds",
        "icon": "[MUSIC]",
        "title": "Music Bed Generator",
        "subtitle": "Generate music for social paid ads",
        "tools": ["Stability Audio 2.5", "Suno V5", "ElevenLabs"],
        "input_type": "prompt",
        "placeholder": "Describe mood, tempo, duration (e.g., upbeat 15s for Instagram)...",
        "color": "#3B82F6"
    },
    {
        "id": "social_assets",
        "icon": "[SHARE]",
        "title": "Social Asset Pack",
        "subtitle": "Generate sized variants for all platforms",
        "tools": ["Stability SD3.5", "Krea AI", "Figma"],
        "input_type": "file",
        "placeholder": "Paste source asset link or describe the campaign...",
        "color": "#10B981"
    },
    {
        "id": "video_from_brief",
        "icon": "[FILM]",
        "title": "Video from Brief",
        "subtitle": "Script -> Storyboard -> Edit pipeline",
        "tools": ["Claude Opus", "Luma Dream Machine", "Runway Gen-3", "After Effects"],
        "input_type": "prompt",
        "placeholder": "Paste creative brief or describe the video concept...",
        "color": "#F59E0B"
    },
    {
        "id": "avatar_presenter",
        "icon": "[USER]",
        "title": "Avatar Presenter",
        "subtitle": "HeyGen clones with brand voice",
        "tools": ["HeyGen", "ElevenLabs"],
        "input_type": "prompt",
        "placeholder": "Paste script or describe the presentation...",
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
        'generated_content': {},
        # Orchestrator state
        'orchestrator_type': 'auto',  # 'auto', 'threadpool', 'claude'
        'active_pipeline': None,
        'pipeline_history': [],
        'task_results': {},
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def get_active_orchestrator() -> BaseOrchestrator:
    """Get orchestrator based on user preference or auto-detect"""
    pref = st.session_state.get('orchestrator_type', 'auto')
    if pref == 'threadpool':
        return ThreadPoolOrchestrator()
    elif pref == 'claude':
        return ClaudeTaskOrchestrator()
    else:
        return get_orchestrator()

def render_pipeline_progress(pipeline: Pipeline):
    """Render visual progress tracker for active pipeline"""
    if not pipeline:
        return

    st.markdown(f"""
    <div style="background:#1a1a2e;border:1px solid #333;border-radius:12px;padding:1rem;margin:1rem 0">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem">
            <strong>[REFRESH] {pipeline.name}</strong>
            <span style="color:#888;font-size:0.85rem">{pipeline.status.value.upper()}</span>
        </div>
        <div style="background:#333;border-radius:4px;height:8px;overflow:hidden">
            <div style="background:linear-gradient(90deg,#9333EA,#06B6D4);height:100%;width:{pipeline.progress()*100}%;transition:width 0.3s"></div>
        </div>
        <div style="color:#888;font-size:0.75rem;margin-top:0.5rem">{int(pipeline.progress()*100)}% complete</div>
    </div>
    """, unsafe_allow_html=True)

    # Task breakdown by stage
    for stage in PipelineStage:
        stage_tasks = pipeline.get_tasks_by_stage(stage)
        if not stage_tasks:
            continue

        with st.expander(f"[OPEN] {stage.value.replace('_', ' ').title()}", expanded=stage_tasks[0].status == TaskStatus.RUNNING):
            for task in stage_tasks:
                status_icon = {
                    TaskStatus.PENDING: "[TIME]",
                    TaskStatus.RUNNING: "[REFRESH]",
                    TaskStatus.COMPLETED: "[OK]",
                    TaskStatus.FAILED: "[X]",
                    TaskStatus.CANCELLED: "[STOP]"
                }.get(task.status, "[?]")

                duration_str = ""
                if task.duration():
                    duration_str = f" ({task.duration():.1f}s)"

                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"{status_icon} **{task.name}** `{task.tool}`{duration_str}")
                with col2:
                    if task.status == TaskStatus.COMPLETED and task.result:
                        st.markdown(":package: Result")

                if task.error:
                    st.error(f"Error: {task.error}")

def render_orchestrator_panel():
    """Render orchestrator controls in sidebar or panel"""
    st.markdown("### [BOLT] Orchestrator")

    # Environment detection
    orch = get_active_orchestrator()
    orch_type = "Claude Agents" if isinstance(orch, ClaudeTaskOrchestrator) else "Thread Pool"
    orch_color = "#9333EA" if isinstance(orch, ClaudeTaskOrchestrator) else "#06B6D4"

    st.markdown(f"""
    <div style="background:#1a1a2e;border:1px solid {orch_color};border-radius:8px;padding:0.75rem;margin-bottom:1rem">
        <span style="color:{orch_color};font-weight:bold">* {orch_type}</span>
        <p style="color:#888;font-size:0.75rem;margin:0.25rem 0 0 0">
            {'Sub-agent parallel execution' if isinstance(orch, ClaudeTaskOrchestrator) else 'ThreadPoolExecutor (5 workers)'}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Override selector
    orch_choice = st.selectbox(
        "Orchestration Mode",
        ["Auto-detect", "Thread Pool", "Claude Agents"],
        index=["auto", "threadpool", "claude"].index(st.session_state.get('orchestrator_type', 'auto')),
        help="Auto-detect uses Claude agents when available, otherwise thread pool"
    )
    st.session_state.orchestrator_type = ["auto", "threadpool", "claude"][["Auto-detect", "Thread Pool", "Claude Agents"].index(orch_choice)]

    # Active pipeline status
    if st.session_state.get('active_pipeline'):
        render_pipeline_progress(st.session_state.active_pipeline)

    # Quick actions
    st.markdown("#### Quick Pipelines")
    if st.button("[VIDEO] Video Pipeline", use_container_width=True, help="Script -> Storyboard -> Video -> Audio (parallel)"):
        st.session_state.show_video_pipeline_modal = True
    if st.button("[IMG] Image Variations", use_container_width=True, help="Generate 3 variations in parallel"):
        st.session_state.show_image_pipeline_modal = True

def check_mcp_status():
    """Check which MCP servers are configured"""
    status = {
        'after-effects': False,
        'adobe-express': False,
        'figma': False,
        'photoshop': False,
        'illustrator': False,
        'premiere': False
    }

    try:
        config_paths = [
            Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json",
            Path.home() / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json"
        ]

        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                        servers = config.get('mcpServers', {})
                        status['after-effects'] = 'after-effects' in servers
                        status['adobe-express'] = 'adobe-express' in servers
                        status['figma'] = 'figma' in servers
                        status['photoshop'] = any('ps-mcp' in str(v) for v in servers.values())
                        status['illustrator'] = any('ai-mcp' in str(v) for v in servers.values())
                        status['premiere'] = any('premiere' in k.lower() for k in servers.keys())
                except (json.JSONDecodeError, IOError, PermissionError):
                    pass
    except Exception:
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
    
    label = {'new': '[SPARK] NEW', 'mcp': '[PLUG] MCP', 'setup': '[WARN] SETUP'}.get(status, '')
    
    return f'<span class="tool-pill {status_class}">{name} {label}</span>'

# ============================================================================
# PAGES
# ============================================================================

def page_lander():
    """Main landing page with workflow selection and goal input"""
    st.markdown('<h1 class="main-header">SOLUS FORGE</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Creative Command Center</p>', unsafe_allow_html=True)

    # Session Goal Input - the key workflow driver
    st.markdown("### What are you working on today?")

    session_goal = st.text_area(
        "Describe your goal or paste a link to your brief/assets",
        placeholder="e.g., Update Q1 email template with new hero images from the Dropbox folder, localize the CTA for APAC markets...",
        height=100,
        key="session_goal_input"
    )

    if session_goal:
        st.session_state.session_goal = session_goal

    st.markdown("---")
    st.markdown("### Or choose a workflow:")

    # Intent Grid - 3 columns
    cols = st.columns(3)
    for i, intent in enumerate(INTENT_MODULES):
        with cols[i % 3]:
            # Card-style button with description
            card_html = f"""
            <div style="background:linear-gradient(135deg, #1a1a2e, #16213e);
                        border:2px solid {intent['color']}40;
                        border-radius:12px;
                        padding:1rem;
                        margin:0.5rem 0;
                        cursor:pointer;
                        transition:all 0.2s ease;">
                <div style="font-size:1.5rem;margin-bottom:0.5rem">{intent['icon']}</div>
                <div style="font-weight:600;color:#fff">{intent['title']}</div>
                <div style="font-size:0.85rem;color:#888;margin-top:0.25rem">{intent['subtitle']}</div>
                <div style="font-size:0.75rem;color:{intent['color']};margin-top:0.5rem">
                    {' + '.join(intent['tools'][:3])}{'...' if len(intent['tools']) > 3 else ''}
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

            if st.button(
                f"Start {intent['title']}",
                key=f"intent_{intent['id']}",
                use_container_width=True,
                type="primary" if i < 3 else "secondary"
            ):
                st.session_state.selected_intent = intent['id']
                st.session_state.current_view = 'workspace'
                st.rerun()

    st.markdown("---")

    # Quick context for new users
    with st.expander("How FORGE works", expanded=False):
        st.markdown("""
        **1. Describe your goal** - Paste a brief, link to assets, or just describe what you need

        **2. Choose a workflow** - Pick the pipeline that matches your task

        **3. FORGE orchestrates** - We route to the right tools (Stability, Runway, ElevenLabs, etc.)

        **4. Review & export** - Get your assets, make tweaks, push to production

        ---

        **Common workflows:**
        - *Email Template* - Drop a Google Doc with annotations, get updated templates
        - *Video Localizer* - Batch process videos with localized end cards
        - *Music Beds* - Generate on-brand audio for social ads
        """)

    # Stack Overview (collapsed)
    with st.expander("Current Stack Status", expanded=False):
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
    """Modular workspace view with goal-driven context"""
    intent = next((i for i in INTENT_MODULES if i['id'] == st.session_state.selected_intent), INTENT_MODULES[0])

    # Header
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if st.button("<- Back"):
            st.session_state.current_view = 'lander'
            st.session_state.selected_intent = None
            st.rerun()
    with col2:
        st.markdown(f"<h2 style='text-align:center'>{intent['icon']} {intent['title']}</h2>", unsafe_allow_html=True)
    with col3:
        st.selectbox("Model", ["Claude Sonnet", "Claude Opus", "Claude Haiku", "GPT-4o"], key="model_select", label_visibility="collapsed")

    # Show session goal if set
    session_goal = st.session_state.get('session_goal', '')
    if session_goal:
        st.info(f"**Goal:** {session_goal[:200]}{'...' if len(session_goal) > 200 else ''}")

    st.markdown("---")

    # Workspace Layout
    col_canvas, col_panel = st.columns([2, 1])

    with col_canvas:
        st.markdown("### [ART] Canvas")

        # Intent-specific input based on input_type
        input_type = intent.get('input_type', 'prompt')
        placeholder = intent.get('placeholder', 'Describe what you need...')

        if input_type == 'file':
            st.markdown(f"**{intent['subtitle']}**")
            file_link = st.text_input("Paste link to source file", placeholder=placeholder, key="file_link_input")
            uploaded = st.file_uploader("Or upload directly", accept_multiple_files=True, key="file_upload")
            if file_link or uploaded:
                st.success("Assets ready - click Generate to process")

        elif input_type == 'batch':
            st.markdown(f"**{intent['subtitle']}**")
            batch_desc = st.text_area("Batch configuration", placeholder=placeholder, height=120, key="batch_input")
            uploaded = st.file_uploader("Upload video assets", accept_multiple_files=True, type=['mp4', 'mov', 'webm'], key="batch_upload")
            if uploaded:
                st.markdown(f"**{len(uploaded)} files ready for processing**")

        else:  # prompt type
            st.markdown(f"**{intent['subtitle']}**")
            prompt_input = st.text_area("Your prompt", placeholder=placeholder, height=120, key="prompt_input")

        # Generate button
        col_gen1, col_gen2 = st.columns(2)
        with col_gen1:
            if st.button("[BOLT] Generate", use_container_width=True, type="primary"):
                st.session_state.generating = True
                st.rerun()
        with col_gen2:
            if st.button("[REFRESH] Clear", use_container_width=True):
                for key in ['file_link_input', 'file_upload', 'batch_input', 'prompt_input']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

        # Intent-specific workspace rendering
        if intent['id'] == 'email_template':
            render_email_template_workspace()
        elif intent['id'] == 'video_localizer':
            render_video_localizer_workspace()
        elif intent['id'] == 'music_beds':
            render_music_beds_workspace()
        elif intent['id'] == 'social_assets':
            render_social_assets_workspace()
        elif intent['id'] == 'video_from_brief':
            render_video_story_workspace()
        elif intent['id'] == 'avatar_presenter':
            render_avatar_workspace()
        else:
            # Fallback for legacy intents
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

    with col_panel:
        # Orchestrator Panel
        render_orchestrator_panel()

        st.markdown("---")

        st.markdown("### [PLUG] Tools")
        for tool in intent['tools']:
            status = get_tool_status(tool)
            dot_class = {'new': 'dot-green', 'active': 'dot-green', 'mcp': 'dot-purple', 'setup': 'dot-yellow'}.get(status, 'dot-gray')
            st.markdown(f'<div class="model-selector"><span class="status-dot {dot_class}"></span>{tool}</div>', unsafe_allow_html=True)

        st.markdown("### [FOLDER] Assets")
        st.file_uploader("Drop files here", accept_multiple_files=True, label_visibility="collapsed", key="panel_upload")

        st.markdown("### [MEMO] Notes")
        st.text_area("Session notes", height=100, label_visibility="collapsed", placeholder="Add notes about this project...", key="session_notes")


def render_email_template_workspace():
    """Email Template Editor workflow"""
    st.markdown("""
    <div style="background:linear-gradient(135deg, #1a1a2e, #2d1f3d);border:2px solid #FF6B35;border-radius:16px;padding:1.5rem;margin:1rem 0">
        <strong>[EMAIL] Email Template Pipeline</strong>
        <p style="color:#888;font-size:0.9rem">Parse Doc -> Extract Copy -> Generate Images -> Compose Template</p>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline steps
    steps = ["Parse Source Doc", "Extract Copy Changes", "Generate New Images", "Compose Template", "Export"]
    current_step = st.session_state.get('email_step', 0)

    cols = st.columns(len(steps))
    for i, step in enumerate(steps):
        with cols[i]:
            if i < current_step:
                st.markdown(f"[OK] {step}")
            elif i == current_step:
                st.markdown(f"[BOLT] **{step}**")
            else:
                st.markdown(f"[ ] {step}")

    # Preview area
    st.markdown("#### Preview")
    preview_area = st.empty()
    preview_area.markdown("""
    <div style="border:1px dashed #444;border-radius:8px;padding:2rem;text-align:center;color:#666">
        Template preview will appear here after processing
    </div>
    """, unsafe_allow_html=True)

def render_video_localizer_workspace():
    """Video Localizer batch processing workflow"""
    st.markdown("""
    <div style="background:linear-gradient(135deg, #1a1a2e, #2d1f3d);border:2px solid #9333EA;border-radius:16px;padding:1.5rem;margin:1rem 0">
        <strong>[VIDEO] Video Localizer Pipeline</strong>
        <p style="color:#888;font-size:0.9rem">Batch Process -> Generate End Cards -> Append -> Export</p>
    </div>
    """, unsafe_allow_html=True)

    # Region configuration
    st.markdown("#### Target Regions")
    col1, col2 = st.columns(2)
    with col1:
        regions = st.multiselect("Select regions", ["APAC", "EMEA", "LATAM", "NA"], default=["APAC"])
    with col2:
        languages = st.multiselect("Languages", ["EN", "ES", "ZH", "JA", "KO", "DE", "FR"], default=["EN"])

    # End card specs
    st.markdown("#### End Card Specs")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.text_input("CTA Text", value="Learn More", key="endcard_cta")
    with col2:
        st.text_input("Logo variant", value="horizontal_white", key="endcard_logo")
    with col3:
        st.number_input("Duration (sec)", value=3, min_value=1, max_value=10, key="endcard_duration")

    # Batch status
    st.markdown("#### Batch Queue")
    st.markdown("""
    <div style="border:1px solid #333;border-radius:8px;padding:1rem">
        <p style="color:#666;margin:0">No videos in queue. Upload files to begin.</p>
    </div>
    """, unsafe_allow_html=True)

def render_music_beds_workspace():
    """Music Bed Generator workflow"""
    st.markdown("""
    <div style="background:linear-gradient(135deg, #1a1a2e, #16213e);border:2px solid #3B82F6;border-radius:16px;padding:1.5rem;margin:1rem 0">
        <strong>[MUSIC] Music Bed Generator</strong>
        <p style="color:#888;font-size:0.9rem">Stability Audio 2.5 + Suno V5 -> Mix -> Export</p>
    </div>
    """, unsafe_allow_html=True)

    # Music parameters
    col1, col2 = st.columns(2)
    with col1:
        st.selectbox("Mood", ["Upbeat", "Calm", "Energetic", "Dramatic", "Playful", "Corporate"], key="music_mood")
        st.selectbox("Genre", ["Electronic", "Acoustic", "Orchestral", "Pop", "Ambient", "Hip-Hop"], key="music_genre")
    with col2:
        st.slider("Tempo (BPM)", 60, 180, 120, key="music_tempo")
        st.slider("Duration (sec)", 5, 60, 15, key="music_duration")

    # Platform presets
    st.markdown("#### Platform Presets")
    preset_cols = st.columns(4)
    presets = [("Instagram", "15s"), ("TikTok", "30s"), ("YouTube", "60s"), ("Stories", "10s")]
    for i, (platform, dur) in enumerate(presets):
        with preset_cols[i]:
            if st.button(f"{platform} ({dur})", key=f"preset_{platform}", use_container_width=True):
                st.session_state.music_duration = int(dur.replace('s', ''))

    # Generated tracks
    st.markdown("#### Generated Tracks")
    st.markdown("""
    <div style="border:1px dashed #444;border-radius:8px;padding:2rem;text-align:center;color:#666">
        Generated music tracks will appear here
    </div>
    """, unsafe_allow_html=True)

def render_social_assets_workspace():
    """Social Asset Pack generator"""
    st.markdown("""
    <div style="background:linear-gradient(135deg, #1a1a2e, #16213e);border:2px solid #10B981;border-radius:16px;padding:1.5rem;margin:1rem 0">
        <strong>[SHARE] Social Asset Pack</strong>
        <p style="color:#888;font-size:0.9rem">Source -> Generate Variants -> Resize -> Export Pack</p>
    </div>
    """, unsafe_allow_html=True)

    # Platform selection
    st.markdown("#### Target Platforms")
    platforms = {
        "Instagram": ["Feed (1080x1080)", "Story (1080x1920)", "Reel (1080x1920)"],
        "Facebook": ["Feed (1200x630)", "Story (1080x1920)", "Ad (1080x1080)"],
        "LinkedIn": ["Feed (1200x627)", "Story (1080x1920)"],
        "Twitter/X": ["Feed (1200x675)", "Header (1500x500)"],
        "TikTok": ["Video (1080x1920)", "Profile (200x200)"]
    }

    selected_platforms = st.multiselect("Select platforms", list(platforms.keys()), default=["Instagram", "Facebook"])

    # Show sizes for selected platforms
    if selected_platforms:
        st.markdown("#### Sizes to generate:")
        for platform in selected_platforms:
            sizes = platforms.get(platform, [])
            st.markdown(f"**{platform}:** {', '.join(sizes)}")

    # Asset grid
    st.markdown("#### Generated Assets")
    st.markdown("""
    <div style="border:1px dashed #444;border-radius:8px;padding:2rem;text-align:center;color:#666">
        Asset variants will appear here after generation
    </div>
    """, unsafe_allow_html=True)

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
    """Video from Story workflow with orchestration support"""
    st.markdown('''
    <div style="background:linear-gradient(135deg, #1a1a2e, #2d1f3d);border:2px solid #9333EA;border-radius:16px;padding:1.5rem;margin:1rem 0">
        <strong>[VIDEO] Video Production Pipeline</strong>
        <p style="color:#888;font-size:0.9rem">Claude -> Krea -> Luma/Runway -> After Effects</p>
    </div>
    ''', unsafe_allow_html=True)

    # Show active pipeline progress if running
    if st.session_state.get('active_pipeline'):
        render_pipeline_progress(st.session_state.active_pipeline)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["[SCRIPT] Script", "[LIST] Shot List", "[CHAR] Characters", "[BOARD] Storyboard", "[TRAIN] Style Training", "[GEN] Generate", "[PIPE] Pipeline"])

    with tab1:
        st.text_area("Script / Treatment", height=200, placeholder="Write your story here... Claude will help with formatting, coverage, and creative development.")
        col1, col2 = st.columns(2)
        with col1:
            st.button("[AI] Claude: Expand & Develop", use_container_width=True)
        with col2:
            st.button("[FORMAT] Format Screenplay", use_container_width=True)

    with tab2:
        # Shot List Ingest Tab
        st.markdown("""
        <div style="background:#1a1a2e;border:1px solid #30363d;border-radius:8px;padding:1rem;margin-bottom:1rem">
            <strong>[LIST] Shot List Ingest</strong>
            <p style="color:#888;font-size:0.85rem;margin:0.5rem 0 0 0">
                Import shot lists from spreadsheets, docs, or paste directly. Auto-parses into structured shots for pipeline execution.
            </p>
        </div>
        """, unsafe_allow_html=True)

        ingest_method = st.radio("Import Method", ["Paste Text", "Upload File", "Link (Dropbox/GDocs)"], horizontal=True)

        if ingest_method == "Paste Text":
            shot_list_text = st.text_area(
                "Paste Shot List",
                height=200,
                placeholder="""Shot 1: Wide establishing - City skyline at dusk, drone push-in
Shot 2: Medium - Hero enters frame left, walks toward camera
Shot 3: Close-up - Hero's face, expression changes from neutral to determined
Shot 4: Insert - Hand grabs door handle
...""",
                key="shot_list_paste"
            )
        elif ingest_method == "Upload File":
            uploaded_file = st.file_uploader("Upload Shot List", type=["csv", "xlsx", "txt", "pdf"], key="shot_list_upload")
            if uploaded_file:
                st.success(f"Uploaded: {uploaded_file.name}")
        else:
            shot_list_link = st.text_input("Dropbox or Google Docs Link", placeholder="https://docs.google.com/spreadsheets/d/... or https://dropbox.com/...")

        st.markdown("---")
        st.markdown("##### Parse Settings")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.selectbox("Shot Format", ["Auto-detect", "Numbered (Shot 1, 2...)", "Timecode (00:00:00)", "Scene/Shot (1A, 1B...)"], key="shot_format")
        with col2:
            st.selectbox("Duration Source", ["From timecode", "Estimate from description", "Manual per shot"], key="duration_source")
        with col3:
            default_duration = st.number_input("Default Duration (sec)", min_value=1, max_value=60, value=5, key="default_shot_duration")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("[PARSE] Parse Shot List", use_container_width=True, type="primary"):
                st.session_state.shot_list_parsed = True
                st.success("Shot list parsed! 12 shots detected.")
        with col2:
            if st.button("[AI] Claude: Enhance Descriptions", use_container_width=True):
                st.info("Enhancing shot descriptions with visual detail...")

        # Show parsed shots preview
        if st.session_state.get('shot_list_parsed'):
            st.markdown("##### Parsed Shots Preview")
            preview_data = [
                {"Shot": "1", "Type": "Wide", "Description": "City skyline at dusk, drone push-in", "Duration": "5s", "Status": "Ready"},
                {"Shot": "2", "Type": "Medium", "Description": "Hero enters frame left", "Duration": "4s", "Status": "Ready"},
                {"Shot": "3", "Type": "Close-up", "Description": "Hero's face, expression shift", "Duration": "3s", "Status": "Ready"},
            ]
            st.dataframe(preview_data, use_container_width=True)
            st.button("[SEND] Send to Storyboard", use_container_width=True)

    with tab3:
        st.text_input("Character Name")
        st.text_area("Character Description", height=100)
        st.button("[GEN] Generate Character Sheet (SD3.5)", use_container_width=True)

    with tab4:
        st.slider("Number of panels", 4, 24, 8)
        st.selectbox("Style", ["Cinematic", "Comic", "Anime", "Realistic"])
        col1, col2 = st.columns(2)
        with col1:
            st.button("[FAST] Iterate (Krea)", use_container_width=True)
        with col2:
            st.button("[FINAL] Final Frames (SD3.5)", use_container_width=True, type="primary")

    with tab5:
        # Visual Style Training Tab
        st.markdown("""
        <div style="background:linear-gradient(135deg, #1a1a2e, #2d1f3d);border:2px solid #F59E0B;border-radius:12px;padding:1rem;margin-bottom:1rem">
            <strong>[TRAIN] Visual Style Training</strong>
            <p style="color:#888;font-size:0.85rem;margin:0.5rem 0 0 0">
                Train custom LoRAs for animation style, character consistency, or motion patterns. Local training on 3090 or cloud via Replicate.
            </p>
        </div>
        """, unsafe_allow_html=True)

        training_type = st.selectbox("Training Type", [
            "Animation Style (movement patterns, timing)",
            "Visual Style (color grading, lighting, texture)",
            "Character Consistency (face, body, clothing)",
            "Motion Transfer (reference video to new subject)",
            "Hybrid (style + character)"
        ], key="training_type")

        st.markdown("---")
        st.markdown("##### Reference Material")

        ref_source = st.radio("Reference Source", ["Upload Images/Video", "Link (Dropbox/Drive)", "Existing Project Assets"], horizontal=True, key="ref_source")

        if ref_source == "Upload Images/Video":
            ref_files = st.file_uploader(
                "Upload Reference Material",
                type=["png", "jpg", "jpeg", "mp4", "mov", "gif"],
                accept_multiple_files=True,
                key="training_refs"
            )
            if ref_files:
                st.success(f"{len(ref_files)} files uploaded")
        elif ref_source == "Link (Dropbox/Drive)":
            st.text_input("Folder Link", placeholder="https://dropbox.com/sh/... or https://drive.google.com/...", key="ref_link")
        else:
            st.selectbox("Select Project", ["Q1 Campaign - Hero Assets", "Brand Refresh 2026", "Product Launch Video"], key="existing_project")

        st.markdown("---")
        st.markdown("##### Training Configuration")

        col1, col2 = st.columns(2)
        with col1:
            st.selectbox("Base Model", [
                "SDXL 1.0 (images)",
                "Stable Video Diffusion (video)",
                "AnimateDiff v3 (motion)",
                "Luma Dream Machine (scenes)"
            ], key="base_model")
            st.slider("Training Steps", 500, 5000, 1500, step=100, key="training_steps")
            st.slider("Learning Rate", 1e-6, 1e-4, 1e-5, format="%.1e", key="learning_rate")

        with col2:
            st.selectbox("Compute Target", [
                "Local (Mac Pro 3090 - 24GB)",
                "Replicate (A100 - fast)",
                "Modal (H100 - fastest)",
                "Queue for overnight"
            ], key="compute_target")
            st.text_input("LoRA Name", placeholder="q1-campaign-style-v1", key="lora_name")
            st.text_input("Trigger Word", placeholder="q1style, campaignlook", key="trigger_word")

        st.markdown("---")
        st.markdown("##### Training Prompts")
        st.text_area(
            "Caption/Prompt Template",
            height=80,
            placeholder="A [subject] in q1style, cinematic lighting, film grain, teal and orange color grade...",
            key="caption_template"
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            auto_caption = st.checkbox("Auto-caption with Claude", value=True, key="auto_caption")
        with col2:
            augment = st.checkbox("Data Augmentation", value=True, key="augment")
        with col3:
            regularization = st.checkbox("Use Regularization Images", value=False, key="regularization")

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("[VALIDATE] Validate Dataset", use_container_width=True):
                st.info("Checking image quality, dimensions, and diversity...")
        with col2:
            if st.button("[TRAIN] Start Training", use_container_width=True, type="primary"):
                st.session_state.training_active = True
                st.success("Training job queued! Estimated time: ~45 minutes")

        # Training progress (if active)
        if st.session_state.get('training_active'):
            st.markdown("##### Training Progress")
            progress_col1, progress_col2 = st.columns([3, 1])
            with progress_col1:
                st.progress(0.35, "Step 525/1500 - Loss: 0.0234")
            with progress_col2:
                st.button("[STOP] Stop", type="secondary")

    with tab6:
        st.selectbox("Video Model", ["Luma Dream Machine (Iteration)", "Runway Gen-3 (Polish)"])
        st.slider("Duration (seconds)", 5, 60, 15)
        st.button("[GEN] Generate Video", use_container_width=True, type="primary")

    with tab7:
st.markdown("""
        <div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:1rem;margin-bottom:1rem">
            <strong>[BOLT] Full Pipeline Execution</strong>
            <p style="color:#888;font-size:0.85rem;margin:0.5rem 0 0 0">
                Run the complete video production workflow with parallel task execution.
                Voice, SFX, and video generation run simultaneously where possible.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Pipeline configuration
        script_input = st.text_area(
            "Script / Concept",
            height=120,
            placeholder="Enter your video concept or script. The pipeline will:\n1. Expand with Claude\n2. Generate storyboard\n3. Create video (Luma -> Runway)\n4. Generate voice & SFX in parallel",
            key="pipeline_script"
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            style = st.selectbox("Visual Style", ["Cinematic", "Documentary", "Anime", "Abstract"], key="pipeline_style")
        with col2:
            duration = st.slider("Duration (sec)", 5, 60, 15, key="pipeline_duration")
        with col3:
            include_audio = st.checkbox("Include Audio", value=True, key="pipeline_audio")

        st.markdown("---")

        # Pipeline visualization
        st.markdown("##### Pipeline Stages")
        stages_col1, stages_col2, stages_col3, stages_col4 = st.columns(4)
        with stages_col1:
            st.markdown("""
            <div style="text-align:center;padding:0.5rem;background:#1a1a2e;border-radius:8px">
                <div style="font-size:1.5rem">[MEMO]</div>
                <div style="font-size:0.75rem;color:#888">Script</div>
                <div style="font-size:0.65rem;color:#666">Claude</div>
            </div>
            """, unsafe_allow_html=True)
        with stages_col2:
            st.markdown("""
            <div style="text-align:center;padding:0.5rem;background:#1a1a2e;border-radius:8px">
                <div style="font-size:1.5rem">[IMG]</div>
                <div style="font-size:0.75rem;color:#888">Storyboard</div>
                <div style="font-size:0.65rem;color:#666">Krea</div>
            </div>
            """, unsafe_allow_html=True)
        with stages_col3:
            st.markdown("""
            <div style="text-align:center;padding:0.5rem;background:#1a1a2e;border-radius:8px">
                <div style="font-size:1.5rem">[CAMERA]</div>
                <div style="font-size:0.75rem;color:#888">Video</div>
                <div style="font-size:0.65rem;color:#666">Luma -> Runway</div>
            </div>
            """, unsafe_allow_html=True)
        with stages_col4:
            st.markdown("""
            <div style="text-align:center;padding:0.5rem;background:#1a1a2e;border-radius:8px">
                <div style="font-size:1.5rem">[SOUND]</div>
                <div style="font-size:0.75rem;color:#888">Audio</div>
                <div style="font-size:0.65rem;color:#666">|| Parallel</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")

        # Execute button
        if st.button("[ROCKET] Run Full Pipeline", type="primary", use_container_width=True, disabled=not script_input):
            if script_input:
                # Create and execute pipeline
                pipeline = create_video_pipeline(script_input, style.lower(), duration)

                # Add audio tasks if enabled
                if include_audio:
                    music_task = Task(
                        name="Background Music",
                        stage=PipelineStage.GENERATION,
                        tool="suno",
                        params={"prompt": f"Music for: {script_input[:50]}", "genre": "cinematic"},
                        dependencies=[pipeline.tasks[0].id]  # After script
                    )
                    pipeline.add_task(music_task)

                st.session_state.active_pipeline = pipeline

                # Show progress
                progress_placeholder = st.empty()
                status_placeholder = st.empty()

                def update_progress(progress: float, task: Task):
                    progress_placeholder.progress(progress, f"Running: {task.name}")
                    status_placeholder.markdown(f"[OK] Completed: **{task.name}** ({task.tool})")

                # Execute with orchestrator
                orch = get_active_orchestrator()
                with st.spinner("Executing pipeline..."):
                    result = orch.execute_pipeline(pipeline, on_progress=update_progress)

                st.session_state.active_pipeline = result
                st.session_state.pipeline_history.append(result)

                if result.status == TaskStatus.COMPLETED:
                    st.success(f"[OK] Pipeline completed! {len(result.tasks)} tasks executed.")
                else:
                    failed = [t for t in result.tasks if t.status == TaskStatus.FAILED]
                    st.error(f"Pipeline finished with {len(failed)} failed tasks.")

                st.rerun()

def render_audio_workspace():
    """Audio production workflow"""
    tab1, tab2, tab3 = st.tabs(["[SPEAK] Voice (ElevenLabs)", "[MUSIC] Music (Suno)", "[SOUND] SFX (Stability)"])
    
    with tab1:
        st.markdown("**Voice: Vivian** (Australian, female)")
        text = st.text_area("Text to speak", value="Welcome to SOLUS FORGE.", height=100)
        api_key = st.text_input("ElevenLabs API Key", type="password")
        if st.button("[SOUND] Generate Speech", type="primary"):
            if api_key and text:
                st.info("Generating speech... (API integration ready)")
    
    with tab2:
        st.text_area("Music prompt", placeholder="Upbeat electronic, female vocals, 120 BPM...", height=100)
        st.selectbox("Genre", ["Electronic", "Pop", "Cinematic", "Lo-Fi", "Rock"])
        st.button("[MUSIC] Generate Music (Suno V5)", use_container_width=True)
    
    with tab3:
        st.markdown('''
        <div class="stability-highlight">
            <strong>Stability Audio 2.5</strong>
            <p style="color:#888;font-size:0.85rem">Professional SFX, soundscapes, and audio inpainting</p>
        </div>
        ''', unsafe_allow_html=True)
        st.text_area("SFX Description", placeholder="Thunder rumbling, rain on window, distant traffic...", height=80)
        st.slider("Duration", 1, 60, 10, format="%d sec")
        st.button("[SOUND] Generate SFX", use_container_width=True, type="primary")

def render_image_workspace():
    """Image generation workflow"""
    col1, col2 = st.columns([2, 1])

    with col1:
        st.text_area("Prompt", height=150, placeholder="Describe the image you want to create...")
        st.text_area("Negative prompt", height=50, placeholder="What to avoid...")

    with col2:
        st.selectbox("Model", ["Stability SD 3.5 Large", "SDXL", "Krea Real-time"])
        st.selectbox("Aspect Ratio", ["1:1", "16:9", "9:16", "4:3", "3:2"])
        st.slider("Steps", 20, 50, 30)
        st.slider("CFG Scale", 1.0, 15.0, 7.5)

    if st.button("[IMG] Generate Image", type="primary", use_container_width=True):
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
    
    st.button("[ART] Generate Asset", type="primary", use_container_width=True)

def render_quickclips_workspace():
    """Quick video clip generation"""
    st.markdown('''
    <div style="background:#1a1a2e;padding:1rem;border-radius:8px;margin-bottom:1rem">
        <strong>[BOLT] Instant Video Generation</strong>
        <p style="color:#888;font-size:0.85rem;margin:0">Direct text-to-video - Luma for iteration, Runway for polish</p>
    </div>
    ''', unsafe_allow_html=True)

    st.text_area("Video prompt", height=100, placeholder="A golden retriever running through autumn leaves in slow motion...")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.selectbox("Model", ["Luma Dream Machine (Fast)", "Runway Gen-3 (Quality)"])
    with col2:
        st.selectbox("Duration", ["4 sec", "8 sec", "16 sec"])
    with col3:
        st.selectbox("Aspect", ["16:9", "9:16", "1:1"])

    st.button("[BOLT] Generate Clip", type="primary", use_container_width=True)

def render_avatar_workspace():
    """Avatar creation workflow"""
    st.markdown('''
    <div style="background:#1a1a2e;padding:1rem;border-radius:8px;margin-bottom:1rem">
        <strong>[USER] HeyGen Avatar Studio</strong>
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
        st.button("[CAMERA] Generate Avatar Video", type="primary", use_container_width=True)

def page_stack():
    """Full stack overview"""
    st.markdown('<h1 class="main-header">:package: Stack Overview</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">SOLUS FORGE v2.0 - Streamlined Creative Stack</p>', unsafe_allow_html=True)
    
    # Changes Summary
    st.markdown("### [REFRESH] What's Changed in v2.0")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### [+] Added / Prioritized")
        st.markdown("""
        - **Luma Dream Machine** - Direct API, fast iteration
        - **Stability SD 3.5** - Pro image generation
        - **Stability Audio 2.5** - SFX & soundscapes
        - **Stability SPAR3D** - Instant 3D from images
        - **Claude** - Script & creative development
        """)

    with col2:
        st.markdown("#### [-] Removed / Optional")
        st.markdown("""
        - ~~CyberFilm SAGA~~ -> *Build pipeline with existing tools*
        - ~~Saga.so~~ -> *Not essential*
        - **Suno V5** -> *Optional (Stability Audio for SFX)*
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
                                <span style="color:{status_color};font-size:0.8rem">* {info['status'].upper()}</span>
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
                            <span style="color:{status_color};font-size:0.8rem">* {info['status'].upper()}</span>
                        </div>
                        ''', unsafe_allow_html=True)
        
        st.markdown("")

def page_settings():
    """Settings and API keys"""
    st.markdown('<h1 class="main-header">:gear: Settings</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["[KEY] API Keys", "[PLUG] MCP Status", "[BOOK] Setup Guide"])
    
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
        
        st.markdown("### Video Generation")
        st.text_input("Luma AI", type="password", key="api_luma")
        
        if st.button("[SAVE] Save API Keys"):
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
            icon = "[OK]" if connected else "[WARN]"
            st.markdown(f"{icon} **{app}**: {'Connected' if connected else 'Not configured'}")
        
        if st.button("[REFRESH] Refresh MCP Status"):
            st.session_state.mcp_status = check_mcp_status()
            st.rerun()
    
    with tab3:
        st.markdown("""
        ### Quick Setup Guide

        **1. Stability AI** (Image, Audio, 3D)
        - Get API key at [stability.ai](https://stability.ai)
        - Free Community License for <$1M revenue
        - Includes SD 3.5, Audio 2.5, SPAR3D

        **2. Luma Dream Machine**
        - Get API key at [lumalabs.ai](https://lumalabs.ai)
        - Credits-based pricing

        **3. Runway Gen-3**
        - Get API key at [runwayml.com](https://runwayml.com)
        - Credits-based pricing

        **4. Adobe MCP Setup**
        ```bash
        cd ~/Desktop/"Solus Forge 1.2"/adobe-mcp
        ./install-adobe-mcp.sh
        ```

        **5. Restart Claude Desktop** after MCP installation
        """)

# ============================================================================
# MAIN
# ============================================================================

def main():
    load_styles()
    init_state()
    
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("## :fire: FORGE v2.0")
        st.markdown("---")
        
        if st.button(":house: Home", use_container_width=True):
            st.session_state.current_view = 'lander'
            st.session_state.selected_intent = None
            st.rerun()
        
        if st.button(":package: Stack", use_container_width=True):
            st.session_state.current_view = 'stack'
            st.rerun()
        
        if st.button(":gear: Settings", use_container_width=True):
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
        :fire: SOLUS FORGE v2.0 |
        <a href="https://stability.ai" target="_blank">Stability AI</a> |
        <a href="https://lumalabs.ai" target="_blank">Luma</a> |
        <a href="https://runwayml.com" target="_blank">Runway</a> |
        Voice: Vivian (ElevenLabs)
    </p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
