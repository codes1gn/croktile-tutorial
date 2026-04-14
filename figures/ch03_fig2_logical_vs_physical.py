"""
Figure 2: Croqtile's Two-Layer Parallelism Model.
Left: logical parallel tree (parallel / foreach — what the programmer writes).
Right: GPU hardware hierarchy (block > warpgroup > warp > thread).
Arrows show the mapping via space specifiers.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from theme import parse_theme
from manim import *

C, THEME = parse_theme()


class LogicalVsPhysical(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text("Logical Parallelism  →  Physical Hardware", font_size=24,
                      color=C["fg"], font="Monospace")
        title.to_edge(UP, buff=0.3)
        self.add(title)

        sep = DashedLine(UP * 2.8, DOWN * 3.5, color=C["dim"], dash_length=0.1)
        self.add(sep)

        # --- Left: Logical side ---
        log_title = Text("What you write", font_size=16, color=C["blue"],
                         font="Monospace")
        log_title.move_to(LEFT * 3.5 + UP * 2.4)
        log_sub = Text("(logical structure)", font_size=12, color=C["fg3"],
                        font="Monospace")
        log_sub.next_to(log_title, DOWN, buff=0.08)
        self.add(log_title, log_sub)

        def log_box(text, color, pos, w=3.0, h=0.5):
            r = Rectangle(width=w, height=h, fill_color=color,
                          fill_opacity=0.15, stroke_color=color, stroke_width=1.5)
            r.move_to(pos)
            t = Text(text, font_size=12, color=color, font="Monospace")
            t.move_to(r)
            return VGroup(r, t)

        lx = LEFT * 3.8
        b1 = log_box("parallel {px,py} by [8,16]", C["orange"], lx + UP * 1.3, w=3.2)
        b2 = log_box("foreach {tile_k} in [16]", C["purple"], lx + UP * 0.4, w=2.8)
        b3 = log_box("parallel {qx,qy} by [16,16]", C["green"], lx + DOWN * 0.5, w=3.2)
        b4 = log_box("foreach k in [16]", C["teal"], lx + DOWN * 1.4, w=2.2)
        b5 = log_box("compute: += a * b", C["fg2"], lx + DOWN * 2.3, w=1.8)

        for b in [b1, b2, b3, b4, b5]:
            self.add(b)

        pairs = [(b1, b2), (b2, b3), (b3, b4), (b4, b5)]
        for top, bot in pairs:
            a = Arrow(top[0].get_bottom(), bot[0].get_top(), buff=0.05,
                      stroke_width=1.5, color=C["fg3"],
                      max_tip_length_to_length_ratio=0.12)
            self.add(a)

        # Place concurrent/sequential labels to the left of boxes
        lbl_p = Text("concurrent", font_size=10, color=C["orange"],
                      font="Monospace").next_to(b1, LEFT, buff=0.1)
        lbl_f = Text("sequential", font_size=10, color=C["purple"],
                      font="Monospace").next_to(b2, LEFT, buff=0.1)
        lbl_p2 = Text("concurrent", font_size=10, color=C["green"],
                       font="Monospace").next_to(b3, LEFT, buff=0.1)
        lbl_f2 = Text("sequential", font_size=10, color=C["teal"],
                       font="Monospace").next_to(b4, LEFT, buff=0.1)
        self.add(lbl_p, lbl_f, lbl_p2, lbl_f2)

        # --- Right: Physical side ---
        phy_title = Text("Where it runs", font_size=16, color=C["green"],
                         font="Monospace")
        phy_title.move_to(RIGHT * 3.5 + UP * 2.4)
        phy_sub = Text("(GPU hardware)", font_size=12, color=C["fg3"],
                        font="Monospace")
        phy_sub.next_to(phy_title, DOWN, buff=0.08)
        self.add(phy_title, phy_sub)

        rx = RIGHT * 3.5

        def hw_box(text, color, pos, w=3.2, h=0.55):
            r = Rectangle(width=w, height=h, fill_color=color,
                          fill_opacity=0.2, stroke_color=color, stroke_width=2)
            r.move_to(pos)
            t = Text(text, font_size=12, color=color, font="Monospace")
            t.move_to(r)
            return VGroup(r, t)

        h1 = hw_box("Grid: 8×16 = 128 blocks", C["orange"], rx + UP * 1.3)
        h2 = hw_box("Shared Memory (per block)", C["blue"], rx + UP * 0.3, w=2.8)
        h3 = hw_box("256 threads per block", C["green"], rx + DOWN * 0.6, w=2.4)
        h4 = hw_box("Registers (per thread)", C["teal"], rx + DOWN * 1.5, w=2.0)

        for h in [h1, h2, h3, h4]:
            self.add(h)

        hw_pairs = [(h1, h2), (h2, h3), (h3, h4)]
        for top, bot in hw_pairs:
            a = Arrow(top[0].get_bottom(), bot[0].get_top(), buff=0.05,
                      stroke_width=1.5, color=C["fg3"],
                      max_tip_length_to_length_ratio=0.12)
            self.add(a)

        # Mapping arrows
        mappings = [
            (b1[0].get_right(), h1[0].get_left(), C["orange"], ": block"),
            (b3[0].get_right(), h3[0].get_left(), C["green"], ": thread"),
        ]
        for src, dst, color, lbl in mappings:
            a = Arrow(src, dst, buff=0.2, stroke_width=2, color=color,
                      max_tip_length_to_length_ratio=0.06)
            self.add(a)
            lt = Text(lbl, font_size=12, color=color, font="Monospace")
            lt.next_to(a, UP, buff=0.05)
            self.add(lt)

        # Bottom caption
        cap = Text(
            "parallel = concurrent (hardware maps it)    foreach = sequential (loop)",
            font_size=12, color=C["fg3"], font="Monospace"
        )
        cap.to_edge(DOWN, buff=0.3)
        self.add(cap)
