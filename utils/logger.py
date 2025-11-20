"""Simple logger for MaintSight operations."""

import sys
from typing import Optional

try:
    from rich.console import Console
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    Console = None
    Text = None


class Logger:
    """Simple logger with emoji support and colored output."""
    
    def __init__(self, name: str):
        """Initialize logger with a name.
        
        Args:
            name: Logger name (usually module or class name)
        """
        self.name = name
        if HAS_RICH and Console is not None:
            self.console = Console()
        else:
            self.console = None
        
    def info(self, message: str, emoji: Optional[str] = None) -> None:
        """Log info message.
        
        Args:
            message: Message to log
            emoji: Optional emoji prefix
        """
        prefix = f"{emoji} " if emoji else ""
        if HAS_RICH and self.console:
            self.console.print(f"{prefix}{message}", style="blue")
        else:
            print(f"{prefix}{message}")
        
    def warn(self, message: str, emoji: Optional[str] = None) -> None:
        """Log warning message.
        
        Args:
            message: Message to log
            emoji: Optional emoji prefix
        """
        prefix = f"{emoji} " if emoji else "⚠️  "
        if HAS_RICH and self.console:
            self.console.print(f"{prefix}{message}", style="yellow")
        else:
            print(f"{prefix}{message}")
        
    def error(self, message: str, emoji: Optional[str] = None) -> None:
        """Log error message.
        
        Args:
            message: Message to log
            emoji: Optional emoji prefix
        """
        prefix = f"{emoji} " if emoji else "❌ "
        if HAS_RICH and self.console:
            # Use regular print to stderr for error messages to avoid rich console file parameter issues
            print(f"{prefix}{message}", file=sys.stderr)
        else:
            print(f"{prefix}{message}", file=sys.stderr)
        
    def success(self, message: str, emoji: Optional[str] = None) -> None:
        """Log success message.
        
        Args:
            message: Message to log
            emoji: Optional emoji prefix
        """
        prefix = f"{emoji} " if emoji else "✅ "
        if HAS_RICH and self.console:
            self.console.print(f"{prefix}{message}", style="green")
        else:
            print(f"{prefix}{message}")
        
    def debug(self, message: str, emoji: Optional[str] = None) -> None:
        """Log debug message.
        
        Args:
            message: Message to log  
            emoji: Optional emoji prefix
        """
        prefix = f"{emoji} " if emoji else "🔍 "
        if HAS_RICH and self.console:
            self.console.print(f"{prefix}[{self.name}] {message}", style="dim")
        else:
            print(f"{prefix}[{self.name}] {message}")