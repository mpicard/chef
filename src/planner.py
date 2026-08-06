from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class PlanNode:
    id: str
    tool: str
    args: dict
    deps: list[str] = field(default_factory=list)
    status: NodeStatus = NodeStatus.PENDING
    result: Any = None
    error: str | None = None
    attempts: int = 0


@dataclass
class PlanDAG:
    nodes: dict[str, PlanNode] = field(default_factory=dict)

    def validate(self) -> None:
        for n in self.nodes.values():
            for d in n.deps:
                if d not in self.nodes:
                    raise ValueError(f"Node {n.id!r} depends on missing node {d!r}")
        # simple depth first search
        visited: set[str] = set()
        stack: set[str] = set()

        def dfs(nid: str) -> None:
            if nid in stack:
                raise ValueError(f"Cycle detected at {nid!r}")
            if nid in visited:
                return
            stack.add(nid)
            for d in self.nodes[nid].deps:
                dfs(d)
            stack.discard(nid)
            visited.add(nid)

        for nid in self.nodes:
            dfs(nid)

    def ready_nodes(self) -> list[PlanNode]:
        """Nodes whose deps are all DONE and are themsevles PENDING"""
        out = []
        for n in self.nodes.values():
            if n.status != NodeStatus.PENDING:
                continue

            if all(self.nodes[d].status == NodeStatus.DONE for d in n.deps):
                out.append(n)

        return out

    def aggregate_node(self) -> PlanNode | None:
        """Return the capstone aggregate_report node.

        Mock plans use id ``aggregate``; real LLM planners often mirror the tool
        name (e.g. ``aggregate_report``). Resolve by tool, not hard-coded id.
        """

        if (
            "aggregate" in self.nodes
            and self.nodes["aggregate"].tool == "aggregate_report"
        ):
            return self.nodes["aggregate"]
        matches = [n for n in self.nodes.values() if n.tool == "aggregate_report"]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"DAG has multiple aggregate_report nodes: {[n.id for n in matches]}"
            )
        return None

    def is_done(self) -> bool:
        return all(
            n.status in (NodeStatus.DONE, NodeStatus.FAILED)
            for n in self.nodes.values()
        )

    def any_failed(self) -> bool:
        return any(n.status == NodeStatus.FAILED for n in self.nodes.values())

    def plot(self, filename: str = "plan_dag.png", title: str = "Plan DAG"):
        """Render the DAG as a left-to-right layered diagram and save it as PNG

        Args:
            filename: output filename for the PNG
            title: Figure title
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_rgb
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
        from matplotlib.path import Path

        # Status → (color, glyph, label). Glyph badge keeps state readable
        # without color (colorblind / grayscale print).
        STATUS_STYLE = {
            NodeStatus.PENDING: ("#898781", "○", "pending"),
            NodeStatus.RUNNING: ("#2a78d6", "▶", "running"),
            NodeStatus.DONE: ("#0ca30c", "✓", "done"),
            NodeStatus.FAILED: ("#d03b3b", "✕", "failed"),
        }
        INK, INK_SOFT, EDGE_GRAY = "#0b0b0b", "#52514e", "#b3b2ab"

        def tint(color: str, strength: float = 0.10) -> tuple:
            """Light wash of a color over white (cards stay readable)."""
            r, g, b = to_rgb(color)
            return tuple(1 - strength * (1 - c) for c in (r, g, b))

        # --- 1. Longest-path layering: wave(n) = 1 + max(wave of deps) ---
        wave: dict[str, int] = {}

        def _wave(nid: str) -> int:
            if nid not in wave:
                deps = self.nodes[nid].deps
                wave[nid] = 1 + max((_wave(d) for d in deps), default=-1)
            return wave[nid]

        for nid in self.nodes:
            _wave(nid)

        n_waves = max(wave.values()) + 1
        columns: list[list[str]] = [[] for _ in range(n_waves)]
        for nid, w in wave.items():
            columns[w].append(nid)

        # Order wave 0 by tool (groups get_population / get_timezone / ...);
        # later waves follow the average position of their deps (fewer crossings)
        columns[0].sort(key=lambda i: (self.nodes[i].tool, i))
        rank = {nid: k for k, nid in enumerate(columns[0])}
        for col in columns[1:]:
            col.sort(
                key=lambda i: (
                    sum(rank.get(d, 0) for d in self.nodes[i].deps)
                    / max(len(self.nodes[i].deps), 1),
                    i,
                )
            )
            rank.update({nid: k for k, nid in enumerate(col)})

        # --- 2. Geometry (data units; aspect locked so cards keep their shape) ---
        CARD_W, CARD_H, DX, DY = 3, 0.78, 4.2, 1.08
        tallest = max(len(col) for col in columns)
        pos: dict[str, tuple[float, float]] = {}
        for w, col in enumerate(columns):
            top = (len(col) - 1) * DY / 2.0  # center each column vertically
            for k, nid in enumerate(col):
                pos[nid] = (w * DX, top - k * DY)

        y_top = (tallest - 1) * DY / 2.0 + CARD_H / 2.0
        fig_w = 1.8 + (n_waves - 1) * DX * 0.72 + CARD_W * 0.72
        fig_h = 2.0 + (tallest - 1) * DY * 0.72 + CARD_H * 0.72
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.set_aspect("equal")
        ax.axis("off")

        # --- 3. Edges: smooth S-curves, recessive gray, under the cards ---
        # Arrivals are spread along the target's left edge (sorted by source
        # height) so heavy fan-ins don't collapse into one point.
        for node in self.nodes.values():
            x1, y1 = pos[node.id]
            incoming = sorted(node.deps, key=lambda d: -pos[d][1])
            spread = min(0.14, CARD_H * 0.7 / max(len(incoming), 1))
            for k, dep in enumerate(incoming):
                x0, y0 = pos[dep]
                y_in = y1 + (len(incoming) - 1) * spread / 2 - k * spread
                p0 = (x0 + CARD_W / 2 + 0.05, y0)
                p3 = (x1 - CARD_W / 2 - 0.16, y_in)
                xm = (p0[0] + p3[0]) / 2.0
                path = Path(
                    [p0, (xm, y0), (xm, y_in), p3],
                    [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4],
                )
                ax.add_patch(
                    FancyArrowPatch(
                        path=path,
                        arrowstyle="-|>",
                        mutation_scale=11,
                        lw=1.1,
                        color=EDGE_GRAY,
                        shrinkA=0,
                        shrinkB=0,
                        zorder=1,
                    )
                )

        # --- 4. Node cards: status-tinted box + glyph badge + id / tool text ---
        for nid, (x, y) in pos.items():
            node = self.nodes[nid]
            color, glyph, _ = STATUS_STYLE[node.status]
            ax.add_patch(
                FancyBboxPatch(
                    (x - CARD_W / 2, y - CARD_H / 2),
                    CARD_W,
                    CARD_H,
                    boxstyle="round,pad=0.04,rounding_size=0.16",
                    facecolor=tint(color),
                    edgecolor=color,
                    lw=1.4,
                    zorder=2,
                )
            )
            bx = x - CARD_W / 2 + 0.34  # glyph badge, left side
            ax.add_patch(
                Circle((bx, y), 0.17, facecolor=color, edgecolor="none", zorder=3)
            )
            ax.text(
                bx,
                y,
                glyph,
                ha="center",
                va="center",
                fontsize=8.5,
                color="white",
                fontweight="bold",
                zorder=4,
            )
            tx = x - CARD_W / 2 + 0.62
            ax.text(
                tx,
                y + 0.15,
                node.id,
                ha="left",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=INK,
                zorder=4,
            )
            ax.text(
                tx,
                y - 0.18,
                node.tool,
                ha="left",
                va="center",
                fontsize=8,
                family="monospace",
                color=INK_SOFT,
                zorder=4,
            )

        # --- 5. Wave headers, title, legend ---
        for w in range(n_waves):
            ax.text(
                w * DX,
                y_top + 0.42,
                f"wave {w}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#898781",
            )
        done = sum(n.status is NodeStatus.DONE for n in self.nodes.values())
        ax.text(
            -CARD_W / 2,
            y_top + 1.15,
            title,
            ha="left",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color=INK,
        )
        ax.text(
            -CARD_W / 2,
            y_top + 0.88,
            f"{len(self.nodes)} nodes · {n_waves} waves · {done} done",
            ha="left",
            va="bottom",
            fontsize=9,
            color=INK_SOFT,
        )

        present = [
            s for s in STATUS_STYLE if any(n.status is s for n in self.nodes.values())
        ]
        handles = [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markersize=8,
                markerfacecolor=STATUS_STYLE[s][0],
                markeredgecolor="none",
                label=f"{STATUS_STYLE[s][1]} {STATUS_STYLE[s][2]}",
            )
            for s in present
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=max(len(handles), 1),
            frameon=False,
            fontsize=9,
            handletextpad=0.3,
            columnspacing=1.6,
        )

        ax.set_xlim(-CARD_W / 2 - 0.4, (n_waves - 1) * DX + CARD_W / 2 + 0.4)
        ax.set_ylim(-y_top - 0.7, y_top + 1.7)
        plt.savefig(
            filename, bbox_inches="tight", dpi=200, facecolor=fig.get_facecolor()
        )
        plt.close(fig)

    @classmethod
    def from_json(cls, data: dict) -> "PlanDAG":
        dag = cls()
        for n in data["nodes"]:
            dag.nodes[n["id"]] = PlanNode(
                id=n["id"],
                tool=n["tool"],
                args=n["args"],
                deps=n.get("deps", []),
            )
        dag.validate()
        return dag


class Planner:
    def plan(self, goal: str) -> PlanDAG:
        pass

    def replan(self, goal: str, failed_dag: PlanDAG) -> PlanDAG:
        pass
