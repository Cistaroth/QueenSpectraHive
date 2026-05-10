import logging
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme
from rich.rule import Rule
from rich.traceback import install as install_rich_traceback

class CustomConsole(Console):
    """Custom console to add a section() method."""
    def section(self, title: str) -> None:
        """
        Print a section header.

        Args:
            title (str): The title of the section.
        Returns:
            None
        """
        self.print()
        self.print(Rule(title, style="cyan"))

console = CustomConsole()

install_rich_traceback(console=console, show_locals=False)

theme = Theme({
    "log.time": "cyan",
})

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            console=console,
            markup=True,
            rich_tracebacks=True,
            omit_repeated_times=False,
        )
    ],
)

logger = logging.getLogger()

