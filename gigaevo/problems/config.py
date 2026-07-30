from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from gigaevo.programs.metrics.context import VALIDITY_KEY, MetricSpec


class ParameterSpec(BaseModel):
    """Specification for a function parameter."""

    name: str = Field(description="Parameter name")
    type_hint: str | None = Field(
        default=None,
        description="Type hint string (e.g., 'np.ndarray', 'dict[str, float]')",
    )
    description: str | None = Field(
        default=None,
        description="Parameter description for docstring",
    )
    default: str | None = Field(
        default=None,
        description="Default value expression (if any)",
    )


class ReturnSpec(BaseModel):
    """Specification for return value."""

    type_hint: str | None = Field(
        default=None,
        description="Return type hint (e.g., 'dict[str, float]')",
    )
    description: str | None = Field(
        default=None,
        description="Human-readable description of return value",
    )
    fields: dict[str, str] | None = Field(
        default=None,
        description="For dict returns: mapping of key -> type/description",
    )


class FunctionSignature(BaseModel):
    """Function signature with structured parameter and return specs."""

    params: list[ParameterSpec] = Field(
        default_factory=list,
        description="List of parameter specifications",
    )
    returns: ReturnSpec | None = Field(
        default=None,
        description="Return value specification",
    )

    def get_param_names(self) -> list[str]:
        """Extract parameter names."""
        return [p.name for p in self.params]

    def get_param_string(self, with_types: bool = True) -> str:
        """Generate function parameter string with optional type hints."""
        parts = []
        for p in self.params:
            if with_types and p.type_hint:
                parts.append(f"{p.name}: {p.type_hint}")
            else:
                parts.append(p.name)
        return ", ".join(parts)


class HelperFunctionSpec(BaseModel):
    """Specification for a helper function stub."""

    name: str = Field(description="Function name")
    description: str = Field(description="What this helper does")
    signature: FunctionSignature = Field(description="Function signature")


class ContextSpec(BaseModel):
    """Specification for build_context() return value."""

    description: str | None = Field(
        default=None,
        description="What this context contains",
    )
    fields: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of context key -> type/description",
    )


class UtilsImportSpec(BaseModel):
    """Specification for importing functions from problem type's utils.py."""

    functions: list[str] = Field(
        description="Function names to import. Use ['*'] for wildcard import."
    )

    @model_validator(mode="after")
    def _validate_functions(self) -> UtilsImportSpec:
        """Validate function list is non-empty and wildcard usage."""
        if not self.functions:
            raise ValueError("functions must contain at least one name or '*'")
        if "*" in self.functions and len(self.functions) > 1:
            raise ValueError("'*' cannot be combined with other function names")
        return self


class UtilsConfig(BaseModel):
    """Configuration for utils imports across generated files."""

    validator: UtilsImportSpec | None = Field(
        default=None,
        description="Utils imports for validate.py",
    )
    helper: UtilsImportSpec | None = Field(
        default=None,
        description="Utils imports for helper.py",
    )
    context: UtilsImportSpec | None = Field(
        default=None,
        description="Utils imports for context.py",
    )
    initial_programs: UtilsImportSpec | None = Field(
        default=None,
        description="Utils imports for initial_programs/*.py",
    )

    @model_validator(mode="before")
    @classmethod
    def _empty_functions_means_no_import(cls, data: object) -> object:
        """Treat ``{"functions": []}`` as "no import" (None).

        LLM generation reliably means "nothing to import here" but writes
        an empty list instead of omitting the field / using null --
        UtilsImportSpec itself correctly rejects an empty list (it's a
        meaningless import spec on its own), so normalize here, before
        pydantic tries to build the nested UtilsImportSpec and rejects it.
        """
        if not isinstance(data, dict):
            return data
        for key in ("validator", "helper", "context", "initial_programs"):
            value = data.get(key)
            if isinstance(value, dict) and not value.get("functions"):
                data[key] = None
        return data


class TaskDescription(BaseModel):
    """Task description with optional hints and metadata."""

    objective: str = Field(
        description="Problem description with objective, rules, goals"
    )
    hints: list[str] | None = Field(
        default=None,
        description="Optional strategy hints",
    )
    constraints: list[str] | None = Field(
        default=None,
        description="List of constraints for the solution",
    )
    failure_modes: list[str] | None = Field(
        default=None,
        description="Common errors to avoid",
    )
    output_shape: str | None = Field(
        default=None,
        description="Expected output shape (e.g., '(11, 2) NumPy array')",
    )
    fitness_goal: str | None = Field(
        default=None,
        description="Fitness target (e.g., 'min_area ≥ 0.0365')",
    )
    complexity_notes: str | None = Field(
        default=None,
        description="Notes about problem complexity and search landscape",
    )
    validation_notes: list[str] | None = Field(
        default=None,
        description="Critical validation and efficiency requirements",
    )


class InitialProgram(BaseModel):
    """Specification for an initial/seed program."""

    name: str = Field(description="Program filename (without .py)")
    description: str = Field(description="Strategy description")


class ProblemConfigValidator:
    """Centralized validation for ProblemConfig."""

    @classmethod
    def validate_all(cls, config: ProblemConfig) -> list[str]:
        """Run all validations, return list of error messages."""
        errors: list[str] = []
        errors.extend(cls._validate_metrics(config))
        return errors

    @staticmethod
    def _validate_metrics(config: ProblemConfig) -> list[str]:
        """Validate metrics structure."""
        errors = []
        if VALIDITY_KEY in config.metrics:
            errors.append(f"'{VALIDITY_KEY}' is auto-generated. Remove it from config.")
        primary_count = sum(1 for s in config.metrics.values() if s.is_primary)
        if primary_count != 1:
            errors.append(f"Exactly one metric must be primary, found {primary_count}")
        return errors



class ProblemConfig(BaseModel):
    """Complete problem specification for scaffolding."""

    name: str = Field(description="Problem directory name")
    description: str = Field(description="Short problem description")

    entrypoint: FunctionSignature
    validation: FunctionSignature

    metrics: dict[str, MetricSpec] = Field(
        description="Metric specifications (is_valid auto-generated, do NOT include)"
    )

    task_description: TaskDescription

    add_context: bool = Field(default=False, description="Generate context.py")
    add_helper: bool = Field(default=False, description="Generate helper.py")

    initial_programs: list[InitialProgram] = Field(
        default_factory=list,
        description="Initial seed programs",
    )

    helper_functions: list[HelperFunctionSpec] | None = Field(
        default=None,
        description="Helper function specifications (if add_helper=True)",
    )
    context_spec: ContextSpec | None = Field(
        default=None,
        description="Context specification (if add_context=True)",
    )
    utils_imports: UtilsConfig | None = Field(
        default=None,
        description="Utils imports configuration for generated files",
    )

    @model_validator(mode="after")
    def _fill_missing_primary_bounds(self) -> ProblemConfig:
        """Auto-fill lower_bound/upper_bound for the primary metric when missing.

        ProblemContext._load_metrics_context requires both bounds on the
        primary metric at run.py time. Confirmed live: the model does not
        reliably self-correct this within TaskBuilderAgent's small retry
        budget (each retry re-generates a whole new config and can
        introduce a *different* missing-bounds metric rather than fixing
        the one flagged) -- it's a purely mechanical property, so auto-fill
        a generous default range instead of spending another LLM round-trip
        on it. 1e5 matches MetricSpec's own MIN/MAX_VALUE_DEFAULT sentinel
        magnitude, so it can never trap the auto-assigned sentinel_value
        inside the bounds (which MetricSpec itself would reject).
        """
        for spec in self.metrics.values():
            if not spec.is_primary:
                continue
            if spec.lower_bound is None:
                spec.lower_bound = 0.0
            if spec.upper_bound is None:
                spec.upper_bound = 1e5
        return self

    @model_validator(mode="after")
    def _drop_bogus_initial_program_utils_imports(self) -> ProblemConfig:
        """Strip seed-program names out of ``utils_imports.initial_programs``.

        Confirmed live: the model sometimes lists the seed programs' own
        names (e.g. ``naive_tim_sort``) as functions to import from the
        shared utils module -- those names were never exported by utils.py,
        they're just what it called its own initial_programs/*.py files.
        Importing them raises ImportError, so every seed program fails
        before ``entrypoint`` even runs. Auto-correct rather than reject +
        retry: this is unambiguous and doesn't need another LLM round-trip.
        """
        spec = self.utils_imports.initial_programs if self.utils_imports else None
        if spec is None or not self.initial_programs:
            return self
        program_names = {p.name for p in self.initial_programs}
        kept = [f for f in spec.functions if f not in program_names and f != "*"]
        if len(kept) != len(spec.functions):
            self.utils_imports.initial_programs = (
                UtilsImportSpec(functions=kept) if kept else None
            )
        return self

    @model_validator(mode="after")
    def _reconcile_context_and_helper_flags(self) -> ProblemConfig:
        """Auto-fix add_context/add_helper vs. their supporting fields.

        Confirmed live: TaskBuilderAgent's retry budget wasn't enough here
        either -- the model sets ``context_spec``/``helper_functions``
        (clear intent to use them) but forgets to flip the matching
        ``add_context``/``add_helper`` flag, or declares
        ``add_context=True`` without adding a ``context`` parameter to
        both signatures. All three are mechanically resolvable without
        another LLM round-trip: flip the flag to match the intent already
        expressed by the data, and append the missing parameter.
        """
        if self.context_spec is not None and not self.add_context:
            self.add_context = True
        if self.helper_functions and not self.add_helper:
            self.add_helper = True

        if self.add_context:
            for sig in (self.entrypoint, self.validation):
                if "context" not in sig.get_param_names():
                    sig.params.append(
                        ParameterSpec(
                            name="context",
                            type_hint="dict",
                            description="Read-only problem context from build_context().",
                        )
                    )
        return self

    @model_validator(mode="after")
    def _run_validations(self) -> ProblemConfig:
        """Run all config validations."""
        errors = ProblemConfigValidator.validate_all(self)
        if errors:
            raise ValueError("\n".join(errors))
        return self
