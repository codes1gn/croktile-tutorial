"""
Figure 1: 2D tensor contraction as a general concept.
A[M,K] × B[K,N] → C[M,N] at tile level, with multiple hardware backends.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from manim import *
from theme import parse_theme

C, THEME = parse_theme()


class TensorContraction(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text(
            "2D Tensor Contraction: C += A × B",
            font_size=22, color=C["fg"], font="Monospace",
        )
        title.to_edge(UP, buff=0.3)
        self.add(title)

        def make_tile(label, w, h, color, pos):
            r = Rectangle(width=w, height=h, fill_color=color,
                          fill_opacity=0.2, stroke_color=color, stroke_width=2)
            t = Text(label, font_size=14, color=color, font="Monospace")
            g = VGroup(r, t)
            t.move_to(r.get_center())
            g.move_to(pos)
            return g

        y_top = 0.8
        a = make_tile("A [M, K]", 1.4, 1.8, C["lhs_c"], LEFT * 3.5 + UP * y_top)
        b = make_tile("B [K, N]", 1.8, 1.4, C["rhs_c"], LEFT * 0.3 + UP * y_top)
        c_tile = make_tile("C [M, N]", 1.8, 1.8, C["out_c"], RIGHT * 3.0 + UP * y_top)

        times = Text("×", font_size=28, color=C["fg"], font="Monospace")
        times.move_to((a.get_right() + b.get_left()) / 2)
        arrow = Arrow(b.get_right() + RIGHT * 0.15, c_tile.get_left() + LEFT * 0.15,
                      buff=0, stroke_width=2.5, color=C["arrow_c"],
                      max_tip_length_to_length_ratio=0.15)
        eq_label = Text("+=", font_size=16, color=C["fg2"], font="Monospace")
        eq_label.next_to(arrow, UP, buff=0.08)

        self.add(a, b, c_tile, times, arrow, eq_label)

        dim_a = Text("M rows, K cols", font_size=10, color=C["dim"], font="Monospace")
        dim_a.next_to(a, DOWN, buff=0.12)
        dim_b = Text("K rows, N cols", font_size=10, color=C["dim"], font="Monospace")
        dim_b.next_to(b, DOWN, buff=0.12)
        dim_c = Text("M rows, N cols", font_size=10, color=C["dim"], font="Monospace")
        dim_c.next_to(c_tile, DOWN, buff=0.12)
        self.add(dim_a, dim_b, dim_c)

        sub_title = Text(
            "Same operation, different hardware implementations",
            font_size=14, color=C["fg3"], font="Monospace",
        )
        sub_title.move_to(DOWN * 0.8)
        self.add(sub_title)

        backends = [
            ("GPU Tensor Core", "16×16×16 FP16", C["green"]),
            ("TPU MXU", "128×128 systolic", C["blue"]),
            ("Intel AMX", "16×64 tiles", C["orange"]),
            ("Custom DSA", "vendor-specific", C["purple"]),
        ]

        x_start = -4.5
        x_step = 3.0
        y_hw = -2.0

        for i, (name, detail, color) in enumerate(backends):
            x = x_start + i * x_step
            box = RoundedRectangle(
                corner_radius=0.1, width=2.6, height=1.1,
                fill_color=color, fill_opacity=0.12,
                stroke_color=color, stroke_width=1.5,
            )
            box.move_to([x, y_hw, 0])
            n = Text(name, font_size=12, color=color, font="Monospace")
            d = Text(detail, font_size=10, color=C["dim"], font="Monospace")
            VGroup(n, d).arrange(DOWN, buff=0.08).move_to(box)
            self.add(box, n, d)

        foot = Text(
            "Croktile's mma syntax targets the abstract operation, not a specific chip.",
            font_size=11, color=C["fg3"], font="Monospace",
        )
        foot.to_edge(DOWN, buff=0.3)
        self.add(foot)
