"""Configuration loading and management."""

from __future__ import annotations
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    """LLM provider configuration."""
    provider: str = "anthropic"  # "openai", "anthropic", "local"
    model: str = "claude-3-5-sonnet-20241022"
    api_key: Optional[str] = None
    timeout: int = 30
    temperature: float = 0.7
    max_tokens: int = 1000


@dataclass
class MemoryConfig:
    """Memory backend configuration."""
    backend: str = "in_memory"  # "in_memory", "sqlite", "mirrorcore"
    database_path: Optional[str] = None
    connection_string: Optional[str] = None


@dataclass
class ToolsConfig:
    """Tools configuration."""
    enabled_tools: list[str] = field(default_factory=lambda: [
        "vision_stub",
        "layout_stub",
        "task_planner",
        "scheduler",
    ])


@dataclass
class ElleConfig:
    """Complete Elle configuration."""
    
    # Profile
    profile: str = "dev"
    
    # Components
    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    
    # Paths
    symbols_dir: Optional[Path] = None
    scenes_dir: Optional[Path] = None
    
    # Logging
    log_level: str = "INFO"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ElleConfig':
        """Create config from dictionary."""
        
        config = cls(
            profile=data.get('profile', 'dev'),
            log_level=data.get('log_level', 'INFO'),
        )
        
        # Parse LLM config
        if 'llm' in data:
            llm_data = data['llm']
            config.llm = LLMConfig(
                provider=llm_data.get('provider', 'anthropic'),
                model=llm_data.get('model', 'claude-3-5-sonnet-20241022'),
                api_key=llm_data.get('api_key') or os.getenv('ANTHROPIC_API_KEY'),
                timeout=llm_data.get('timeout', 30),
                temperature=llm_data.get('temperature', 0.7),
                max_tokens=llm_data.get('max_tokens', 1000),
            )
        
        # Parse memory config
        if 'memory' in data:
            mem_data = data['memory']
            config.memory = MemoryConfig(
                backend=mem_data.get('backend', 'in_memory'),
                database_path=mem_data.get('database_path'),
                connection_string=mem_data.get('connection_string'),
            )
        
        # Parse tools config
        if 'tools' in data:
            config.tools = ToolsConfig(
                enabled_tools=data['tools'].get('enabled_tools', [])
            )
        
        # Parse paths
        if 'symbols_dir' in data:
            config.symbols_dir = Path(data['symbols_dir'])
        if 'scenes_dir' in data:
            config.scenes_dir = Path(data['scenes_dir'])
        
        return config
    
    @classmethod
    def from_yaml(cls, path: Path) -> 'ElleConfig':
        """Load config from YAML file."""
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def load_profile(cls, profile: str = "dev") -> 'ElleConfig':
        """
        Load config for a specific profile.
        
        Looks for config file in:
        1. ELLE_CONFIG env var
        2. ./elle_config.yml
        3. ~/.elle/config.yml
        4. Built-in defaults
        """
        
        # Check env var
        env_path = os.getenv('ELLE_CONFIG')
        if env_path and Path(env_path).exists():
            config = cls.from_yaml(Path(env_path))
            config.profile = profile
            return config
        
        # Check local dir
        local_path = Path('./elle_config.yml')
        if local_path.exists():
            config = cls.from_yaml(local_path)
            config.profile = profile
            return config
        
        # Check home dir
        home_path = Path.home() / '.elle' / 'config.yml'
        if home_path.exists():
            config = cls.from_yaml(home_path)
            config.profile = profile
            return config
        
        # Use defaults
        config = cls(profile=profile)
        
        # Adjust defaults by profile
        if profile == "farm":
            config.llm.provider = "anthropic"
            config.memory.backend = "sqlite"
            config.tools.enabled_tools = [
                "vision_real",
                "layout_mc",
                "task_planner",
                "scheduler",
            ]
        elif profile == "studio":
            config.llm.provider = "openai"
            config.memory.backend = "mirrorcore"
        
        return config
