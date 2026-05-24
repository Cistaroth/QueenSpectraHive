from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class ModelPipelineStepInputs:
    """Abstract class to define the inputs of a step."""

    name: str
    dependency: str | int | None = None


@dataclass(frozen=True)
class ModelPipelineStepOutputs:
    """Abstract class to define the outputs of a step."""

    name: str
    output: Any

class ModelPipelineStep(ABC):
    """Abstract class to define a step."""

    name: str
    inputs: set[str]
    outputs: set[str]

    def __init__(self) -> None:
        """Initialize the step.

        Args:
            None
        Returns:
            None
        """
        self._dependency_map: dict[str, str | int | None] = {}

    def _normalize_dependency(
        self,
        parameter_names: (
            list[str] | str | dict[str, str | int | None] | None
        ),
        dependency: str | int | None,
        kwargs: dict[str, str | int | None],
    ) -> dict[str, str | int | None]:
        """
        Normalize the dependency of a parameter.
        
        Args:
            parameter_names: The parameter names. Can be a list, a single
                name, a mapping from name to dependency, or None.
            dependency: The dependency to assign to each name in
                ``parameter_names`` when it is a list or single name.
            kwargs: Alternative way of providing a name-to-dependency
                mapping.
        Returns:
            dict[str, str | int | None]: A mapping from parameter name to
            dependency.
        """

        if isinstance(parameter_names, dict):
            dependency_map = parameter_names
        elif kwargs:
            if parameter_names is not None:
                raise ValueError(
                    f"[{self.name}] Cannot specify both parameter_names "
                    f"and kwargs"
                )
            dependency_map = kwargs
        elif parameter_names is None:
            raise ValueError(
                f"[{self.name}] Must specify parameter_names"
            )
        elif isinstance(parameter_names, list):
            dependency_map = {
                parameter_name: dependency
                for parameter_name in parameter_names
            }
        else:
            dependency_map = {parameter_names: dependency}

        return dependency_map
    
    def set_dependency(
        self,
        parameter_names: (
            list[str] | str | dict[str, str | int | None] | None
        ) = None,
        dependency: str | int | None = None,
        /,
        **kwargs: str | int | None,
    ) -> ModelPipelineStep:
        """Set the dependency of a parameter.

        Examples of supported calls:

        step.set_dependency("dataframe", -1)
        step.set_dependency(["x", "y"], -1)
        step.set_dependency(model="tune", scaler="tune", x_test=-2)
        step.set_dependency({"model": "tune", "x_test": -2})

        Args:
            parameter_names: The parameter names. Can be a list, a single
                name, a mapping from name to dependency, or None.
            dependency: The dependency to assign to each name in
                ``parameter_names`` when it is a list or single name.
            **kwargs: Alternative way of providing a name-to-dependency
                mapping.
        Returns:
            ModelPipelineStep: The step.
        """
        dependency_map = self._normalize_dependency(
            parameter_names=parameter_names,
            dependency=dependency,
            kwargs=kwargs,
        )
        

        # Insert into dependency map
        for name, dep in dependency_map.items():
            if name not in self.inputs:
                raise ValueError(
                    f"[{self.name}] {name} is not a valid input"
                )
            self._dependency_map[name] = dep

        return self

    @property
    def input_overview(self) -> list[ModelPipelineStepInputs]:
        """Return an overview of the inputs.

        Args:
            None
        Returns:
            list[ModelPipelineStepInputs]: Overview of the inputs.
        """
        return [
            ModelPipelineStepInputs(
                name=parameter_name,
                dependency=self._dependency_map.get(parameter_name),
            )
            for parameter_name in self.inputs
        ]

    
    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> dict[str, Any] | None:
        """Run the step. Should be implemented by subclasses."""

    def __str__(self) -> str:
        """Return the name of the step."""
        return self.name


class ModelPipeline:
    """Class for model pipeline."""

    def __init__(
        self,
        steps: (
            dict[str | int, ModelPipelineStep]
            | Sequence[ModelPipelineStep]
            | None
        ) = None,
    ) -> None:
        """Initialize the model pipeline class.

        Args:
            steps: The steps of the model pipeline.
        Returns:
            None
        """

        if steps is None:
            steps = []
        
        # Insert steps and their index and name mappings
        if isinstance(steps, list):
            self._steps = steps
            self._steps_name_map = {str(idx): idx for idx in range(len(steps))}
        elif isinstance(steps, dict):
            self._steps = list(steps.values())
            self._steps_name_map = {
                name: idx for idx, name in enumerate(steps.keys())
            }
        else:
            raise TypeError(
                f"[{self.__class__.__name__}] Steps must be a list or "
                f"dict, got {type(steps).__name__}"
            )

        self._step_labels = list(self._steps_name_map.keys())

        # Contexts - predefined and results
        self._predefined_context: dict[int, dict[str, Any]]= {}
        self._results_context: dict[int, dict[str, Any]] = {}

    def add_context(
        self,
        step: str | int,
        context: dict[str, Any],
    ) -> ModelPipeline:
        """Add predefined context to the model pipeline.

        Context is passed to the run function of the step as kwargs.
        Only usable for predefined contexts; result contexts are
        handled automatically.

        Args:
            step: The step to add the context to.
            context: The context to add.
        Returns:
            ModelPipeline: The model pipeline.
        """

        idx = self._resolve_step(step=step, source="Add_context")
        self._predefined_context[idx] = context
        return self

    def _resolve_step(
        self,
        step: str | int,
        source: str
    ) -> int:
        """
        Resolve a step to its index.
        
        Args:
            step: The step to resolve.
            source: The source of the step.
        Returns:
            int: The index of the step.
        """

        cls_name = self.__class__.__name__

        # If step is an integer
        if isinstance(step, int):
            if step < 0:
                raise ValueError(
                    f"[{cls_name}] Negative index {step} not allowed "
                    f"at {source}"
                )
            if step >= len(self._steps):
                raise ValueError(
                    f"[{cls_name}] Index {step} exceeds the number of "
                    f"steps {len(self._steps)} at {source}"
                )
            return step

        # If step is a string
        if isinstance(step, str):
            if step not in self._steps_name_map:
                raise ValueError(
                    f"[{cls_name}] Step '{step}' not found at {source}"
                )
            return self._steps_name_map[step]

        raise TypeError(
            f"[{cls_name}] Step must be str or int, got "
            f"{type(step).__name__} at {source}"
        )

    def _resolve_dependency(
        self,
        dependency: str | int | None,
        parameter: str,
        index: int,
    ) -> int | None:
        """Resolve a dependency to the providing step's index.

        Resolution rules:
            1. Explicit dependency (str or int) -> that step.
            2. None, but previous step has this output -> previous step.
            3. None otherwise -> predefined context (return None).

        Args:
            dependency: The dependency of the parameter.
            parameter: The name of the parameter.
            index: The index of the current step.
        Returns:
            int | None: The index of the step that provides the
            parameter, or None for the predefined context.
        """
        cls_name = self.__class__.__name__

        # If dependency is None, check if parameter is in predefined context
        # Else check if parameter is in previous step
        if dependency is None:
            if parameter in self._predefined_context.get(index, {}):
                return None
            if (
                index > 0
                and parameter in self._steps[index - 1].outputs
            ):
                return index - 1
            return None
        
        source = self._step_labels[index]
        
        # If dependency is a string
        if isinstance(dependency, str):
            if dependency not in self._steps_name_map:
                raise ValueError(
                    f"[{cls_name}] Step '{dependency}' not found as "
                    f"dependency for parameter '{parameter}' at "
                    f"{source}"
                )
            return self._steps_name_map[dependency]

        # If dependency is an integer, handle absolute
        # and relative dependencies
        if isinstance(dependency, int):
            if dependency >= len(self._steps):
                raise ValueError(
                    f"[{cls_name}] Index for dependency "
                    f"'{dependency}' for parameter '{parameter}' at "
                    f"{source} exceeds the number of steps "
                    f"{len(self._steps)}"
                )
            if dependency < 0:
                resolved = index + dependency
                if resolved < 0:
                    raise ValueError(
                        f"[{cls_name}] Relative dependency "
                        f"{dependency} for parameter '{parameter}' at "
                        f"{source} refers to a step before start of "
                        f"pipeline"
                    )
                return resolved
            return dependency

        raise TypeError(
            f"[{cls_name}] Dependency must be str or int, got "
            f"{type(dependency).__name__} at {source}"
        )

    def _get_parameter(
        self,
        dependency: int | None,
        parameter: str,
        index: int,
    ) -> Any:
        """Get the parameter from the context.

        Args:
            dependency: The index of the step that provides the
                parameter, or None for predefined context.
            parameter: The name of the parameter.
            index: The index of the current step.
        Returns:
            Any: The parameter.
        """
        # If dependency is None, use predefined context
        # Else use results context
        if dependency is None:
            context = self._predefined_context.get(index, {})
        else:
            context = self._results_context.get(dependency, {})

        # Check if parameter is in context
        if parameter not in context:
            source = f"step {self._step_labels[index]}"
            if dependency is None:
                context_message = "predefined context"
            else:
                context_message = (
                    f"results context of step {self._step_labels[dependency]}"
                )
            raise ValueError(
                f"[{self.__class__.__name__}] Parameter '{parameter}' "
                f"for {source} not found in {context_message}"
            )

        return context[parameter]

    def _validate_steps(self) -> None:
        """Validate the steps of the model pipeline.

        Check if steps are defined and run a preliminary check that
        parameters will be available at runtime.

        Args:
            None
        Returns:
            None
        """
        cls_name = self.__class__.__name__

        # Check if steps are defined
        if len(self._steps) == 0:
            raise ValueError(f"[{cls_name}] No steps defined")
        
        # Initialize prefined context and results context overview
        predefined_overview = {
            idx: parameters.keys()
            for idx, parameters in self._predefined_context.items()
        }
        results_overview = {}

        # Check if parameters are available
        for idx, step in enumerate(self._steps):
            step_name = self._step_labels[idx]
            for parameter in step.input_overview:
                dependency = self._resolve_dependency(
                    dependency=parameter.dependency,
                    parameter=parameter.name,
                    index=idx,
                )

                if dependency is None:
                    # Check if parameter is in predefined context
                    if parameter.name not in predefined_overview.get(
                        idx, {}
                    ):
                        raise ValueError(
                            f"[{cls_name}] Parameter "
                            f"'{parameter.name}' for step "
                            f"{step_name}:{idx} is not in predefined "
                            f"context"
                        )
                else:
                    # Check if parameter is in results context
                    if parameter.name not in results_overview.get(
                        dependency, set()
                    ):
                        raise ValueError(
                            f"[{cls_name}] Parameter "
                            f"'{parameter.name}' for step "
                            f"{step_name}:{idx} is not in results "
                            f"context"
                        )

            # Add outputs to results overview
            results_overview[idx] = step.outputs

    def run(self) -> list[ModelPipelineStepOutputs]:
        """Run the model pipeline.

        Returns:
            list[ModelPipelineStepOutputs]: The outputs/history of the
            model pipeline.
        """
        # Validate steps
        self._validate_steps()

        # Run steps
        self._results_context = {}
        for idx, step in enumerate(self._steps):
            kwargs: dict[str, Any] = {}
            for parameter in step.input_overview:
                dependency = self._resolve_dependency(
                    dependency=parameter.dependency,
                    parameter=parameter.name,
                    index=idx,
                )
                kwargs[parameter.name] = self._get_parameter(
                    dependency=dependency,
                    parameter=parameter.name,
                    index=idx,
                )

            result = step.run(**kwargs)
            self._results_context[idx] = (
                result if result is not None else {}
            )

        # Compile history
        history = [
            ModelPipelineStepOutputs(
                name=str(label),
                output=self._results_context[idx],
            )
            for idx, label in enumerate(self._step_labels)
        ]
        
        # Reset results context
        self._results_context = {}

        return history