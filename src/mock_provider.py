import json
import re

from llm_provider import LLMProvider

CITY_FACTS = {
    "paris": {"population": 2_100_000, "timezone": "CET", "country": "France"},
    "tokyo": {"population": 13_960_000, "timezone": "JST", "country": "Japan"},
    "new york": {"population": 8_335_000, "timezone": "EST", "country": "USA"},
    "london": {"population": 8_982_000, "timezone": "GMT", "country": "UK"},
    "sydney": {"population": 5_312_000, "timezone": "AEDT", "country": "Australia"},
}


class MockProvider(LLMProvider):
    def complete(self, system: str, user: str, role: str = "default") -> str:
        # Planner: produce a DAG decomposition
        if role == "planner":
            cities = self._extract_cities(user)
            return json.dumps(self._build_plan(cities))

        # Re-planner: handle failures and produce amended DAG
        if role == "replanner":
            cities = self._extract_cities(user)

            # TODO: why is this included?
            # Extract failed cities from the user message
            failed_cities = [c for c in cities if c not in CITY_FACTS]

            # Build plan only for known cities
            known_cities = [c for c in cities if c in CITY_FACTS]
            if known_cities:
                return json.dumps(self._build_plan(known_cities))
            else:
                # No valid cities, return minimal plan
                return json.dumps({"nodes": []})

        # Summarizer worker: produce a one-line city summary
        if role == "summarizer":
            city = self._extract_city(user)
            facts = CITY_FACTS.get(city.lower(), {})
            country = facts.get("country", "its country")
            pop = facts.get("population", 0)
            return json.dumps(
                {
                    "summary": (
                        f"{city.title()} is a major city in {country} "
                        f"with roughly {pop / 1e6:.1f}M residents."
                    )
                }
            )

        # Aggregator: compose a markdown report from the gathered facts
        if role == "aggregator":
            return json.dumps(
                {
                    "report": self._compose_report(user),
                }
            )

        # Critic: judge the report against acceptance criteria.
        # The user string contains "GOAL:\n...\n\nREPORT:\n...". We extract the
        # report portion and check it mentions multiple cities and has numeric
        # evidence (e.g., population figures).
        if role == "critic":
            report_part = user.split("REPORT:", 1)[-1].lower()
            cities_mentioned = sum(1 for c in CITY_FACTS if c in report_part)
            has_numbers = bool(re.search(r"\d{3,}", report_part))
            passes = cities_mentioned >= 2 and has_numbers
            return json.dumps(
                {
                    "pass": passes,
                    "reason": (
                        f"Report covers {cities_mentioned} cities with data."
                        if passes
                        else "Report is missing cities or quantitative data."
                    ),
                }
            )

        return json.dumps({"error": f"unknown role {role!r}"})

    # --- helpers below are mock-only; a real LLM wouldn't need them ---
    def _extract_cities(self, text: str) -> list[str]:
        found = [c for c in CITY_FACTS if c in text.lower()]
        # Also check for unknown cities mentioned in the text
        for word in ["atlantis", "unknown", "narnia"]:
            if word in text.lower():
                found.append(word)
        return found or ["paris", "tokyo", "new york"]

    def _extract_city(self, text: str) -> str:
        for c in CITY_FACTS:
            if c in text.lower():
                return c
        return "paris"

    def _build_plan(self, cities: list[str]) -> dict:
        nodes = []
        fetch_ids_by_city = {}
        for city in cities:
            pop_id = f"pop_{city.replace(' ', '_')}"
            tz_id = f"tz_{city.replace(' ', '_')}"
            sum_id = f"sum_{city.replace(' ', '_')}"
            nodes.append(
                {
                    "id": pop_id,
                    "tool": "get_population",
                    "args": {"city": city},
                    "deps": [],
                }
            )
            nodes.append(
                {
                    "id": tz_id,
                    "tool": "get_timezone",
                    "args": {"city": city},
                    "deps": [],
                }
            )
            nodes.append(
                {
                    "id": sum_id,
                    "tool": "summarize_city",
                    "args": {"city": city},
                    "deps": [],
                }
            )
            fetch_ids_by_city[city] = [pop_id, tz_id, sum_id]
        all_ids = [nid for ids in fetch_ids_by_city.values() for nid in ids]
        nodes.append(
            {
                "id": "aggregate",
                "tool": "aggregate_report",
                "args": {"cities": cities},
                "deps": all_ids,
            }
        )
        return {"nodes": nodes}

    def _compose_report(self, user: str) -> str:
        # Extract the facts the aggregator was handed (stored as JSON inside user prompt)
        lines = ["# City Comparison Report", ""]
        for c, f in CITY_FACTS.items():
            if c in user.lower():
                lines.append(
                    f"- **{c.title()}** ({f['country']}): "
                    f"{f['population']:,} people, timezone {f['timezone']}."
                )
        return "\n".join(lines)
