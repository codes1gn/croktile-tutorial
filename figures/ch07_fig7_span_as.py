"""
Figure 7 (ch07): span_as — 1D buffer reinterpreted as 2D matrix.
Shows a flat linear buffer being viewed as a [rows, cols] matrix
without any data copy.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from theme import parse_theme
from manim import *

C, THEME = parse_theme()


class SpanAs(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text("span_as: zero-copy shape reinterpretation", font_size=18, color=C["fg"], font="Monospace")
        title.to_edge(UP, buff=0.3)
        self.add(title)

        cell_w = 0.35
        cell_h = 0.45

        # Top: 1D buffer (16 elements in a row)
        n_elems = 16
        top_origin = LEFT * (n_elems * cell_w / 2) + UP * 1.7

        top_label = Text("1D buffer (16 elements)", font_size=12, color=C["fg2"], font="Monospace")
        top_label.move_to(UP * 2.3)
        self.add(top_label)

        for i in range(n_elems):
            row_color = C["blue"] if i < 4 else C["green"] if i < 8 else C["orange"] if i < 12 else C["purple"]
            sq = Rectangle(width=cell_w, height=cell_h, fill_color=row_color, fill_opacity=0.25,
                           stroke_color=row_color, stroke_width=1)
            sq.move_to(top_origin + RIGHT * i * cell_w)
            self.add(sq)
            t = Text(str(i), font_size=12, color=C["fg"], font="Monospace")
            t.move_to(sq)
            self.add(t)

        # Arrow
        arrow = Arrow(DOWN * (-0.9) + UP * 0.5, DOWN * 0.4,
                      color=C["arrow"], stroke_width=2, buff=0.1, max_tip_length_to_length_ratio=0.13)
        self.add(arrow)

        arrow_label = Text(".span_as([4, 4])", font_size=12, color=C["arrow"], font="Monospace")
        arrow_label.next_to(arrow, RIGHT, buff=0.3)
        self.add(arrow_label)

        # Bottom: 2D matrix (4x4)
        bot_origin = LEFT * 0.65 + DOWN * 1.2
        bot_label = Text("2D view [4, 4]", font_size=12, color=C["fg2"], font="Monospace")
        bot_label.move_to(DOWN * 0.6)
        self.add(bot_label)

        row_colors = [C["blue"], C["green"], C["orange"], C["purple"]]
        for r in range(4):
            for c in range(4):
                idx = r * 4 + c
                sq = Rectangle(width=cell_w * 1.2, height=cell_h, fill_color=row_colors[r], fill_opacity=0.25,
                               stroke_color=row_colors[r], stroke_width=1)
                sq.move_to(bot_origin + RIGHT * c * cell_w * 1.2 + DOWN * r * cell_h)
                self.add(sq)
                t = Text(str(idx), font_size=12, color=C["fg"], font="Monospace")
                t.move_to(sq)
                self.add(t)

        note = Text("same memory, different logical shape -- no copy", font_size=13, color=C["fg3"], font="Monospace")
        note.to_edge(DOWN, buff=0.3)
        self.add(note)
