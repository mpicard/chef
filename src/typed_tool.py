from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError


@dataclass
class TypedTool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[..., Any]
    cost_hint: float = 0.0

    def schema(self) -> dict:
        """Shape expected for OpenAI-API-compatible tool schemas."""

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.args_model.model_json_schema(),
        }

    def run(self, raw_args: dict) -> Any:
        args, err = self.validate_args(raw_args)

        if err is not None:
            raise err
        return self.fn(**args.model_dump())

    def validate_args(self, raw_args: dict) -> tuple[BaseModel | None, str | None]:
        try:
            return self.args_model(**raw_args), None
        except ValidationError as e:
            return None, f"ValidationError: {e.errors()}"
