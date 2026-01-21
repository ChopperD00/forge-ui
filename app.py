"""
SOLUS FORGE v2.0 - Creative Command Center
Updated: 2026-01-21
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
    page_icon="\ud83d\udd25",
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
                "desc": "Image \u2192 3D mesh (<1 sec)",
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
        "icon": "\ud83c\udfa8",
        "title": "Brand Assets",
        "subtitle": "Emails, social, brand guides",
        "tools": ["Stability SD3.5", "Krea AI", "Figma", "Illustrator"],
        "color": "#FF6B35"
    },
    {
        "id": "video_story",
        "icon": "\ud83c\udfac",
        "title": "Video from Story",
        "subtitle": "Script \u2192 Storyboard \u2192 Edit",
        "tools": ["Claude Opus", "Krea AI", "Luma Dream Machine", "Runway Gen-3", "After Effects"],
        "color": "#9333EA"
    },
    {
        "id": "audio",
        "icon": "\ud83c\udfb5",
        "title": "Audio Production",
        "subtitle": "Music, VO, SFX",
        "tools": ["ElevenLabs", "Stability Audio 2.5", "Suno V5"],
        "color": "#3B82F6"
    },
    {
        "id": "image_gen",
        "icon": "\ud83d\uddbc\ufe0f",
        "title": "Image Generation",
        "subtitle": "SD3.5, Krea",
        "tools": ["Stability SD3.5", "Krea AI"],
        "color": "#10B981"
    },
    {
        "id": "quick_clips",
        "icon": "\u26a1",
        "title": "Quick Clips",
        "subtitle": "Instant video generation",
        "tools": ["Luma Dream Machine", "Runway Gen-3"],
        "color": "#F59E0B"
    },
    {
        "id": "avatar",
        "icon": "\ud83d\udc64",
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
            <strong>\ud83d\udd04 {pipeline.name}</strong>
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

        with st.expander(f"\ud83d\udcc2 {stage.value.replace('_', ' ').title()}", expanded=stage_tasks[0].status == TaskStatus.RUNNING):
            for task in stage_tasks:
                status_icon = {
                    TaskStatus.PENDING: "\u23f3",
                    TaskStatus.RUNNING: "\ud83d\udd04",
                    TaskStatus.COMPLETED: "\u2705",
                    TaskStatus.FAILED: "\u274c",
                    TaskStatus.CANCELLED: "\u26d4"
                }.get(task.status, "\u2753")

                duration_str = ""
                if task.duration():
                    duration_str = f" ({task.duration():.1f}s)"

                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"{status_icon} **{task.name}** `{task.tool}`{duration_str}")
                with col2:
                    if task.status == TaskStatus.COMPLETED and task.result:
                        st.markdown("\ud83d\udce6 Result")

                if task.error:
                    st.error(f"Error: {task.error}")

def render_orchestrator_panel():
    """Render orchestrator controls in sidebar or panel"""
    st.markdown("### \u26a1 Orchestrator")

    # Environment detection
    orch = get_active_orchestrator()
    orch_type = "Claude Agents" if isinstance(orch, ClaudeTaskOrchestrator) else "Thread Pool"
    orch_color = "#9333EA" if isinstance(orch, ClaudeTaskOrchestrator) else "#06B6D4"

    st.markdown(f"""
    <div style="background:#1a1a2e;border:1px solid {orch_color};border-radius:8px;padding:0.75rem;margin-bottom:1rem">
        <span style="color:{orch_color};font-weight:bold">\u25cf {orch_type}</span>
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
    if st.button("\ud83c\udfac Video Pipeline", use_container_width=True, help="Script \u2192 Storyboard \u2192 Video \u2192 Audio (parallel)"):
        st.session_state.show_video_pipeline_modal = True
    if st.button("\ud83d\uddbc\ufe0f Image Variations", use_container_width=True, help="Generate 3 variations in parallel"):
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
    
    label = {'new': '\u2728 NEW', 'mcp': '\ud83d\udd0c MCP', 'setup': '\u26a0\ufe0f SETUP'}.get(status, '')
    
    return f'<span class="tool-pill {status_class}">{name} {label}</span>'

# ============================================================================
# PAGES
# ============================================================================

def page_lander():
    """Main landing page with intent selection"""
    st.markdown('<h1 class="main-header">\ud83d\udd25 SOLUS FORGE</h1>', unsafe_allow_html=True)
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
        <div class="stability-highlight">
            <h3>\ud83c\udfaf Stability AI Suite</h3>
            <p style="color:#aaa;margin:0.5rem 0">Multimodal generation platform</p>
            <ul style="color:#888;font-size:0.9rem">
                <li>SD 3.5 Large - pro image generation</li>
                <li>Stable Audio 2.5 - SFX & soundscapes</li>
                <li>SPAR3D - image to 3D in <1 second</li>
            </ul>
            <p style="color:#3B82F6;font-size:0.85rem;margin-top:0.75rem">Free Community License (<$1M rev)</p>
        </div>
        ''', unsafe_allow_html=True)

    with col2:
        st.markdown('''
        <div style="background:linear-gradient(135deg, #1a1a2e, #2d1f3d);border:2px solid #9333EA;border-radius:16px;padding:1.5rem;margin:1rem 0">
            <h3>\ud83c\udfac Video Pipeline</h3>
            <p style="color:#aaa;margin:0.5rem 0">Direct API control</p>
            <ul style="color:#888;font-size:0.9rem">
                <li>Luma Dream Machine - fast iteration</li>
                <li>Runway Gen-3 - polish & final</li>
                <li>Claude - script & creative dev</li>
            </ul>
            <p style="color:#9333EA;font-size:0.85rem;margin-top:0.75rem">Your tools, your control</p>
        </div>
        ''', unsafe_allow_html=True)
    
    # Quick Stack Overview
    st.markdown("### \ud83d\udce6 Current Stack")
    
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
        if st.button("\u2190 Back"):
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
        st.markdown("### \ud83c\udfa8 Canvas")
        
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
        # Orchestrator Panel
        render_orchestrator_panel()

        st.markdown("---")

        st.markdown("### \ud83e\uddf0 Tools")
        for tool in intent['tools']:
            status = get_tool_status(tool)
            dot_class = {'new': 'dot-green', 'active': 'dot-green', 'mcp': 'dot-purple', 'setup': 'dot-yellow'}.get(status, 'dot-gray')
            st.markdown(f'<div class="model-selector"><span class="status-dot {dot_class}"></span>{tool}</div>', unsafe_allow_html=True)

        st.markdown("### \ud83d\udcc1 Assets")
        st.file_uploader("Drop files here", accept_multiple_files=True, label_visibility="collapsed")

        st.markdown("### \ud83d\udcdd Notes")
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
    """Video from Story workflow with orchestration support"""
    st.markdown('''
    <div style="background:linear-gradient(135deg, #1a1a2e, #2d1f3d);border:2px solid #9333EA;border-radius:16px;padding:1.5rem;margin:1rem 0">
        <strong>\ud83c\udfac Video Production Pipeline</strong>
        <p style="color:#888;font-size:0.9rem">Claude \u2192 Krea \u2192 Luma/Runway \u2192 After Effects</p>
    </div>
    ''', unsafe_allow_html=True)

    # Show active pipeline progress if running
    if st.session_state.get('active_pipeline'):
        render_pipeline_progress(st.session_state.active_pipeline)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["\ud83d\udcdd Script", "\ud83d\udc65 Characters", "\ud83d\uddbc\ufe0f Storyboard", "\ud83c\udfa5 Generate", "\u26a1 Pipeline"])

    with tab1:
        st.text_area("Script / Treatment", height=200, placeholder="Write your story here... Claude will help with formatting, coverage, and creative development.")
        col1, col2 = st.columns(2)
        with col1:
            st.button("\u2728 Claude: Expand & Develop", use_container_width=True)
        with col2:
            st.button("\ud83d\udcd6 Format Screenplay", use_container_width=True)

    with tab2:
        st.text_input("Character Name")
        st.text_area("Character Description", height=100)
        st.button("\ud83c\udfa8 Generate Character Sheet (SD3.5)", use_container_width=True)

    with tab3:
        st.slider("Number of panels", 4, 24, 8)
        st.selectbox("Style", ["Cinematic", "Comic", "Anime", "Realistic"])
        col1, col2 = st.columns(2)
        with col1:
            st.button("\u26a1 Iterate (Krea)", use_container_width=True)
        with col2:
            st.button("\ud83d\uddbc\ufe0f Final Frames (SD3.5)", use_container_width=True, type="primary")

    with tab4:
        st.selectbox("Video Model", ["Luma Dream Machine (Iteration)", "Runway Gen-3 (Polish)"])
        st.slider("Duration (seconds)", 5, 60, 15)
        st.button("\ud83c\udfac Generate Video", use_container_width=True, type="primary")

    with tab5:
        st.markdown("""
        <div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:1rem;margin-bottom:1rem">
            <strong>\u26a1 Full Pipeline Execution</strong>
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
            placeholder="Enter your video concept or script. The pipeline will:\n1. Expand with Claude\n2. Generate storyboard\n3. Create video (Luma \u2192 Runway)\n4. Generate voice & SFX in parallel",
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
                <div style="font-size:1.5rem">\ud83d\udcdd</div>
                <div style="font-size:0.75rem;color:#888">Script</div>
                <div style="font-size:0.65rem;color:#666">Claude</div>
            </div>
            """, unsafe_allow_html=True)
        with stages_col2:
            st.markdown("""
            <div style="text-align:center;padding:0.5rem;background:#1a1a2e;border-radius:8px">
                <div style="font-size:1.5rem">\ud83d\uddbc\ufe0f</div>
                <div style="font-size:0.75rem;color:#888">Storyboard</div>
                <div style="font-size:0.65rem;color:#666">Krea</div>
            </div>
            """, unsafe_allow_html=True)
        with stages_col3:
            st.markdown("""
            <div style="text-align:center;padding:0.5rem;background:#1a1a2e;border-radius:8px">
                <div style="font-size:1.5rem">\ud83c\udfa5</div>
                <div style="font-size:0.75rem;color:#888">Video</div>
                <div style="font-size:0.65rem;color:#666">Luma \u2192 Runway</div>
            </div>
            """, unsafe_allow_html=True)
        with stages_col4:
            st.markdown("""
            <div style="text-align:center;padding:0.5rem;background:#1a1a2e;border-radius:8px">
                <div style="font-size:1.5rem">\ud83d\udd0a</div>
                <div style="font-size:0.75rem;color:#888">Audio</div>
                <div style="font-size:0.65rem;color:#666">\u2225 Parallel</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")

        # Execute button
        if st.button("\ud83d\ude80 Run Full Pipeline", type="primary", use_container_width=True, disabled=not script_input):
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
                    status_placeholder.markdown(f"\u2705 Completed: **{task.name}** ({task.tool})")

                # Execute with orchestrator
                orch = get_active_orchestrator()
                with st.spinner("Executing pipeline..."):
                    result = orch.execute_pipeline(pipeline, on_progress=update_progress)

                st.session_state.active_pipeline = result
                st.session_state.pipeline_history.append(result)

                if result.status == TaskStatus.COMPLETED:
                    st.success(f"\u2705 Pipeline completed! {len(result.tasks)} tasks executed.")
                else:
                    failed = [t for t in result.tasks if t.status == TaskStatus.FAILED]
                    st.error(f"Pipeline finished with {len(failed)} failed tasks.")

                st.rerun()

def render_audio_workspace():
    """Audio production workflow"""
    tab1, tab2, tab3 = st.tabs(["\ud83d\udde3\ufe0f Voice (ElevenLabs)", "\ud83c\udfb5 Music (Suno)", "\ud83d\udd0a SFX (Stability)"])
    
    with tab1:
        st.markdown("**Voice: Vivian** (Australian, female)")
        text = st.text_area("Text to speak", value="Welcome to SOLUS FORGE.", height=100)
        api_key = st.text_input("ElevenLabs API Key", type="password")
        if st.button("\ud83d\udd0a Generate Speech", type="primary"):
            if api_key and text:
                st.info("Generating speech... (API integration ready)")
    
    with tab2:
        st.text_area("Music prompt", placeholder="Upbeat electronic, female vocals, 120 BPM...", height=100)
        st.selectbox("Genre", ["Electronic", "Pop", "Cinematic", "Lo-Fi", "Rock"])
        st.button("\ud83c\udfb5 Generate Music (Suno V5)", use_container_width=True)
    
    with tab3:
        st.markdown('''
        <div class="stability-highlight">
            <strong>Stability Audio 2.5</strong>
            <p style="color:#888;font-size:0.85rem">Professional SFX, soundscapes, and audio inpainting</p>
        </div>
        ''', unsafe_allow_html=True)
        st.text_area("SFX Description", placeholder="Thunder rumbling, rain on window, distant traffic...", height=80)
        st.slider("Duration", 1, 60, 10, format="%d sec")
        st.button("\ud83d\udd0a Generate SFX", use_container_width=True, type="primary")

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

    if st.button("\ud83d\uddbc\ufe0f Generate Image", type="primary", use_container_width=True):
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
    
    st.button("\ud83c\udfa8 Generate Asset", type="primary", use_container_width=True)

def render_quickclips_workspace():
    """Quick video clip generation"""
    st.markdown('''
    <div style="background:#1a1a2e;padding:1rem;border-radius:8px;margin-bottom:1rem">
        <strong>\u26a1 Instant Video Generation</strong>
        <p style="color:#888;font-size:0.85rem;margin:0">Direct text-to-video \u2014 Luma for iteration, Runway for polish</p>
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

    st.button("\u26a1 Generate Clip", type="primary", use_container_width=True)

def render_avatar_workspace():
    """Avatar creation workflow"""
    st.markdown('''
    <div style="background:#1a1a2e;padding:1rem;border-radius:8px;margin-bottom:1rem">
        <strong>\ud83d\udc64 HeyGen Avatar Studio</strong>
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
        st.button("\ud83c\udfa5 Generate Avatar Video", type="primary", use_container_width=True)

def page_stack():
    """Full stack overview"""
    st.markdown('<h1 class="main-header">\ud83d\udce6 Stack Overview</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">SOLUS FORGE v2.0 - Streamlined Creative Stack</p>', unsafe_allow_html=True)
    
    # Changes Summary
    st.markdown("### \ud83d\udd04 What's Changed in v2.0")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### \u2795 Added / Prioritized")
        st.markdown("""
        - **Luma Dream Machine** - Direct API, fast iteration
        - **Stability SD 3.5** - Pro image generation
        - **Stability Audio 2.5** - SFX & soundscapes
        - **Stability SPAR3D** - Instant 3D from images
        - **Claude** - Script & creative development
        """)

    with col2:
        st.markdown("#### \u2796 Removed / Optional")
        st.markdown("""
        - ~~CyberFilm SAGA~~ \u2192 *Build pipeline with existing tools*
        - ~~Saga.so~~ \u2192 *Not essential*
        - **Suno V5** \u2192 *Optional (Stability Audio for SFX)*
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
                                <span style="color:{status_color};font-size:0.8rem">\u25cf {info['status'].upper()}</span>
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
                            <span style="color:{status_color};font-size:0.8rem">\u25cf {info['status'].upper()}</span>
                        </div>
                        ''', unsafe_allow_html=True)
        
        st.markdown("")

def page_settings():
    """Settings and API keys"""
    st.markdown('<h1 class="main-header">\u2699\ufe0f Settings</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["\ud83d\udd11 API Keys", "\ud83d\udd0c MCP Status", "\ud83d\udcd6 Setup Guide"])
    
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
        
        if st.button("\ud83d\udcbe Save API Keys"):
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
            icon = "\u2705" if connected else "\u26a0\ufe0f"
            st.markdown(f"{icon} **{app}**: {'Connected' if connected else 'Not configured'}")
        
        if st.button("\ud83d\udd04 Refresh MCP Status"):
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
        st.markdown("## \ud83d\udd25 FORGE v2.0")
        st.markdown("---")
        
        if st.button("\ud83c\udfe0 Home", use_container_width=True):
            st.session_state.current_view = 'lander'
            st.session_state.selected_intent = None
            st.rerun()
        
        if st.button("\ud83d\udce6 Stack", use_container_width=True):
            st.session_state.current_view = 'stack'
            st.rerun()
        
        if st.button("\u2699\ufe0f Settings", use_container_width=True):
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
        \ud83d\udd25 SOLUS FORGE v2.0 |
        <a href="https://stability.ai" target="_blank">Stability AI</a> |
        <a href="https://lumalabs.ai" target="_blank">Luma</a> |
        <a href="https://runwayml.com" target="_blank">Runway</a> |
        Voice: Vivian (ElevenLabs)
    </p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
