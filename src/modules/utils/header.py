from rich.panel import Panel
from rich.align import Align

from pipeline import ModelPipelineStep
from config import config
from logger import console

class HeaderModule(ModelPipelineStep):
    name = "Header"
    inputs = {}
    outputs = {}

    def __init__(
        self,
        task: str = "",
    ) -> None:
        """
        Initialize the header module.

        Args:
            task: The task of the model pipeline.
        Returns:
            None
        """
        self._task = f" - {task}" if task else ""

        super().__init__()

    def run(self) -> None:
        """
        Print the header

        Args:
            None
        Returns:
            None
        """
        
        console.print(Panel(
            Align.center(
                "[bold cyan]" + config.APP_NAME + self._task + "[/bold cyan]"
            ),
            border_style="cyan",
            padding=(1, 2),
        ))

