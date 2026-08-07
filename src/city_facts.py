from dataclasses import dataclass

from tools_registry import (
    AggregateArgs,
    CityArgs,
    aggregate_report,
    get_population,
    get_timezone,
    summarize_city,
)
from typed_tool import TypedTool


@dataclass
class GetPopulation(TypedTool):
    name = "get_population"
    description = "Get the population of a city"
    args_model = CityArgs
    fn = get_population
    cost_hint = 0.1


@dataclass
class GetTimezone(TypedTool):
    name = "get_timezone"
    description = "Get the timezone of a city"
    args_model = CityArgs
    fn = get_timezone
    cost_hint = 0.1


@dataclass
class SummarizeCity(TypedTool):
    name = "summarize_city"
    description = "Summarize a city with LLM"
    args_model = CityArgs
    fn = summarize_city
    cost_hint = 1.0


@dataclass
class AggregateReport(TypedTool):
    name = "aggregate_report"
    description = "Aggregate a report of a city with LLM"
    args_model = AggregateArgs
    fn = aggregate_report
    cost_hint = 2.0
