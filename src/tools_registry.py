import json
from typing import Any

from pydantic import BaseModel, Field

from llm import llm
from llm_provider import safe_json_loads
from mock_provider import CITY_FACTS
from typed_tool import TypedTool


class CityArgs(BaseModel):
    city: str = Field(..., description="Name of the city, e.g. 'Paris'.")


class AggregateArgs(BaseModel):
    cities: list[str] = Field(..., description="Cities to include in the report.")


def _fact(city: str, key: str) -> Any:
    facts = CITY_FACTS.get(city.lower())
    if facts is None:
        raise KeyError(f"Unknown city: {city!r}")
    return facts[key]


def get_population(city: str) -> int:
    return _fact(city, "population")


def get_timezone(city: str) -> str:
    return _fact(city, "timezone")


def summarise_city(city: str) -> str:
    raw = llm.complete(
        system='You summarise a city in one sentence. Return JSON: {"summary": "..."}',
        user=f"City: {city}",
        role="summarizer",
    )
    data = safe_json_loads(raw)
    return data.get("summary", f"{city.title()} is a notable city.")


def aggregate_report(cities: list[str]) -> str:
    # Also an LLM-backed tool; the aggregator composes the final report.
    payload = {c: CITY_FACTS.get(c.lower(), {}) for c in cities}
    raw = llm.complete(
        system='Compose a markdown report. Return JSON: {"report": "..."}',
        user=f"Facts: {json.dumps(payload)}",
        role="aggregator",
    )
    data = safe_json_loads(raw)
    return data.get("report", "")


TOOLS: dict[str, TypedTool] = {
    "get_population": TypedTool(
        name="get_population",
        description="Look up a city's population.",
        args_model=CityArgs,
        fn=get_population,
        cost_hint=0.1,
    ),
    "get_timezone": TypedTool(
        name="get_timezone",
        description="Look up a city's timezone.",
        args_model=CityArgs,
        fn=get_timezone,
        cost_hint=0.1,
    ),
    "summarize_city": TypedTool(
        name="summarize_city",
        description="Produce a one-sentence narrative summary of a city.",
        args_model=CityArgs,
        fn=summarise_city,
        cost_hint=1.0,  # LLM call → costlier
    ),
    "aggregate_report": TypedTool(
        name="aggregate_report",
        description="Compose the final markdown comparison report.",
        args_model=AggregateArgs,
        fn=aggregate_report,
        cost_hint=2.0,
    ),
}
