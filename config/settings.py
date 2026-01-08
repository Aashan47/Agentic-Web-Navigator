"""
Configuration settings for the Agentic Web Navigator.
Uses Pydantic Settings for environment variable management.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal
from pathlib import Path


class BrowserSettings(BaseSettings):
    """Browser-specific configuration."""

    headless: bool = Field(default=False, description="Run browser in headless mode")
    viewport_width: int = Field(default=1280, description="Browser viewport width")
    viewport_height: int = Field(default=720, description="Browser viewport height")
    timeout_ms: int = Field(default=30000, description="Default timeout in milliseconds")
    slow_mo: int = Field(default=100, description="Slow down operations by this amount (ms)")
    user_agent: str = Field(
        default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        description="Custom user agent string"
    )

    class Config:
        env_prefix = "BROWSER_"


class LLMSettings(BaseSettings):
    """LLM configuration for Gemini API."""

    google_api_key: str = Field(default="", description="Google Generative AI API key")
    model_name: str = Field(default="gemini-1.5-pro", description="Model to use")
    temperature: float = Field(default=0.1, description="Sampling temperature")
    max_output_tokens: int = Field(default=4096, description="Maximum output tokens")
    top_p: float = Field(default=0.95, description="Top-p sampling")

    class Config:
        env_prefix = "LLM_"


class DOMProcessorSettings(BaseSettings):
    """DOM processing configuration."""

    max_elements: int = Field(default=500, description="Maximum elements to extract")
    max_text_length: int = Field(default=100, description="Max text content length per element")
    include_aria_labels: bool = Field(default=True, description="Include ARIA labels")
    include_data_attributes: bool = Field(default=False, description="Include data-* attributes")

    # Elements to always exclude from DOM tree
    excluded_tags: list[str] = Field(
        default=[
            "script", "style", "noscript", "svg", "path", "meta",
            "link", "head", "title", "br", "hr"
        ],
        description="HTML tags to exclude"
    )

    # Interactive elements to prioritize
    interactive_tags: list[str] = Field(
        default=[
            "a", "button", "input", "select", "textarea", "option",
            "label", "form", "details", "summary"
        ],
        description="Interactive HTML tags to prioritize"
    )

    class Config:
        env_prefix = "DOM_"


class AgentSettings(BaseSettings):
    """Agent behavior configuration."""

    max_steps: int = Field(default=50, description="Maximum steps before stopping")
    max_retries: int = Field(default=3, description="Max retries per action")
    retry_delay_ms: int = Field(default=1000, description="Delay between retries")
    screenshot_quality: int = Field(default=80, description="Screenshot JPEG quality (0-100)")
    action_delay_ms: int = Field(default=500, description="Delay after each action")

    class Config:
        env_prefix = "AGENT_"


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    log_to_file: bool = Field(default=True, description="Enable file logging")
    log_dir: Path = Field(default=Path("logs"), description="Log directory")
    rich_tracebacks: bool = Field(default=True, description="Use Rich for tracebacks")

    class Config:
        env_prefix = "LOG_"


class Settings(BaseSettings):
    """Main settings aggregator."""

    browser: BrowserSettings = Field(default_factory=BrowserSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    dom: DOMProcessorSettings = Field(default_factory=DOMProcessorSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
