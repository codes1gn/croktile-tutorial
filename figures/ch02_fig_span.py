"""
Figure: span(i) — picking one dimension from a tensor's shape.
Shows a multi-dimensional tensor and how span(0), span(1) extract individual dims.
"""
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from theme import parse_theme

from manim import *

C, THEME = parse_theme()


class SpanDimension(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text("span(i) — Picking One Dimension", font_size=26,
                      color=C["fg"], font="Monospace")
        title.to_edge(UP, buff=0.3)
        self.add(title)

        # 2D tensor grid
        rows, cols = 4, 8
        cw, ch = 0.5, 0.45
        grid_ox = -cols * cw / 2 + cw / 2 - 1.5
        grid_oy = 1.2

        tensor_lbl = Text("matrix [M, N]  =  [4, 8]", font_size=14,
                           color=C["fg2"], font="Monospace")
        tensor_lbl.move_to([-1.5, grid_oy + rows * ch / 2 + 0.35, 0])
        self.add(tensor_lbl)

        grid = VGroup()
        for r in range(rows):
            for c in range(cols):
                rect = Rectangle(width=cw - 0.04, height=ch - 0.04,
                                 fill_color=C["fill"], fill_opacity=0.3,
                                 stroke_color=C["dim"], stroke_width=0.6)
                x = grid_ox + c * cw
                y = grid_oy - r * ch
                rect.move_to([x, y, 0])
                grid.add(rect)
        self.add(grid)

        # dim 0 brace (rows)
        row_group = VGroup()
        for r in range(rows):
            rect = Rectangle(width=0.1, height=ch - 0.04,
                              fill_opacity=0)
            rect.move_to([grid_ox - cw / 2 - 0.1, grid_oy - r * ch, 0])
            row_group.add(rect)

        brace_rows = Brace(VGroup(*[grid[r * cols] for r in range(rows)]),
                            LEFT, buff=0.15, color=C["blue"])
        brace_rows_lbl = Text("span(0) = M = 4", font_size=13,
                                color=C["blue"], font="Monospace")
        brace_rows_lbl.next_to(brace_rows, LEFT, buff=0.15)
        self.add(brace_rows, brace_rows_lbl)

        # dim 1 brace (cols)
        brace_cols = Brace(VGroup(*[grid[c] for c in range(cols)]),
                            UP, buff=0.1, color=C["orange"])
        brace_cols_lbl = Text("span(1) = N = 8", font_size=13,
                                color=C["orange"], font="Monospace")
        brace_cols_lbl.next_to(brace_cols, UP, buff=0.1)
        self.add(brace_cols, brace_cols_lbl)

        # .span (full shape)
        span_full = Text("lhs.span  →  [M, N] = [4, 8]  (entire shape)",
                          font_size=13, color=C["green"], font="Monospace")
        span_full.move_to(RIGHT * 3 + UP * 1.2)
        self.add(span_full)

        # Use cases
        use_y = -2.0
        use_box = Rectangle(width=10, height=2.6, fill_color=C["fill"],
                              fill_opacity=0.2, stroke_color=C["dim"], stroke_width=1)
        use_box.move_to([0, use_y, 0])

        use_title = Text("Common Patterns", font_size=16, color=C["fg"], font="Monospace")
        use_title.move_to([0, use_y + 1.1, 0])

        examples = [
            ("s32 [lhs.span] output;", "copy full shape → output is [4, 8]", C["green"]),
            ("s32 [lhs.span(0), rhs.span(1)] result;", "pick M from lhs, N from rhs → matmul output", C["blue"]),
            ("parallel tile by lhs.span(0)", "tile along first axis → 4 tiles", C["blue"]),
        ]

        for i, (code, desc, color) in enumerate(examples):
            c = Text(code, font_size=12, color=color, font="Monospace")
            c.move_to([0, use_y + 0.65 - i * 0.7, 0])
            d = Text(desc, font_size=12, color=C["fg3"], font="Monospace")
            d.move_to([0, use_y + 0.35 - i * 0.7, 0])
            self.add(c, d)

        self.add(use_box, use_title)
