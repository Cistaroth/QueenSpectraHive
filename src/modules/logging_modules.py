from rich.panel import Panel
from rich.align import Align

from pipeline import ModelPipelineStep
from config import config
from logger import console

class HeaderModule(ModelPipelineStep):
    name = "Header"
    inputs = {}
    outputs = {}

    def run(self) -> None:
        """
        Print the header

        Args:
            None
        Returns:
            None
        """
        console.print(Panel(
            Align.center("[bold cyan]" + config.NAME + "[/bold cyan]"),
            border_style="cyan",
            padding=(1, 2),
        ))

    