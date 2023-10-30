from __future__ import annotations

from io import StringIO

from matplotlib.figure import Figure


def fig_to_html(fig: Figure) -> str:
    buffer = StringIO()
    fig.savefig(buffer, format="svg")
    return buffer.getvalue()
