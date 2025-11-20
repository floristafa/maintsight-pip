"""Simple logger for MaintSight operations."""

import sys
from typing import Optional
from rich.console import Console
from rich.text import Text


class Logger:
    """Simple logger with emoji support and colored output."""
    
    def __init__(self, name: str):
        """Initialize logger with a name.
        
        Args:
            name: Logger name (usually module or class name)
        """
        self.name = name
        self.console = Console()
        
    def info(self, message: str, emoji: Optional[str] = None) -> None:
        """Log info message.
        
        Args:
            message: Message to log
            emoji: Optional emoji prefix
        """
        prefix = f"{emoji} " if emoji else ""
        self.console.print(f"{prefix}{message}", style="blue")
        
    def warn(self, message: str, emoji: Optional[str] = None) -> None:
        """Log warning message.
        
        Args:
            message: Message to log
            emoji: Optional emoji prefix
        """
        prefix = f"{emoji} " if emoji else "⚠️  "
        self.console.print(f"{prefix}{message}", style="yellow")
        
    def error(self, message: str, emoji: Optional[str] = None) -> None:
        """Log error message.
        
        Args:
            message: Message to log
            emoji: Optional emoji prefix
        """
        prefix = f"{emoji} " if emoji else "❌ "
        # Use regular print to stderr for error messages to avoid rich console file parameter issues
        print(f"{prefix}{message}", file=sys.stderr)
        
    def success(self, message: str, emoji: Optional[str] = None) -> None:
        """Log success message.
        
        Args:
            message: Message to log
            emoji: Optional emoji prefix
        """
        prefix = f"{emoji} " if emoji else "✅ "
        self.console.print(f"{prefix}{message}", style="green")
        
    def debug(self, message: str, emoji: Optional[str] = None) -> None:
        """Log debug message.
        
        Args:
            message: Message to log  
            emoji: Optional emoji prefix
        """
        prefix = f"{emoji} " if emoji else "🔍 "
        self.console.print(f"{prefix}[{self.name}] {message}", style="dim")