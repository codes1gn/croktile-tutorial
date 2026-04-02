"""
Ch07 Fig8: Expressiveness vs Performance.
Three concentric circles:
  Outer (largest): CUDA expressible range — wide but includes many slow patterns.
  Middle: Performance sweet spot — coalesced, conflict-free, aligned.
  Inner (smallest): Croqtile range — fully contains the sweet spot; always fast.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from theme import parse_theme
from manim import *

C, THEME = parse_theme()


class ExpressivenessVsPerformance(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text(
            "Expressiveness vs Performance",
            font_size=28, color=C["fg"], font="Monospace",
        )
        title.to_edge(UP, buff=0.28)
        self.add(title)

        sub = Text(
            "data movement patterns: who can express what?",
            font_size=15, color=C["fg2"], font="Monospace",
        )
        sub.next_to(title, DOWN, buff=0.1)
        self.add(sub)

        center = DOWN * 0.25

        # Outer circle: CUDA
        cuda_circle = Circle(
            radius=2.8, color=C["orange"], stroke_width=3,
            fill_color=C["orange"], fill_opacity=0.05,
        )
        cuda_circle.move_to(center)
        self.add(cuda_circle)

        cuda_lbl = Text(
            "CUDA expressible range",
            font_size=17, color=C["orange"], font="Monospace",
        )
        cuda_lbl.move_to(cuda_circle.get_top() + DOWN * 0.35)
        self.add(cuda_lbl)

        # Middle circle: performance sweet spot
        sweet_circle = Circle(
            radius=1.7, color=C["green"], stroke_width=3,
            fill_color=C["green"], fill_opacity=0.1,
        )
        sweet_circle.move_to(center)
        self.add(sweet_circle)

        sweet_lbl = Text(
            "Performance sweet spot",
            font_size=16, color=C["green"], font="Monospace",
        )
        sweet_lbl.move_to(sweet_circle.get_top() + DOWN * 0.32)
        self.add(sweet_lbl)

        sweet_sub = Text(
            "coalesced \u00b7 conflict-free \u00b7 aligned swizzle",
            font_size=12, color=C["fg3"], font="Monospace",
        )
        sweet_sub.move_to(sweet_circle.get_center() + UP * 0.85)
        self.add(sweet_sub)

        # Inner circle: Croqtile — fully contains sweet spot
        croq_circle = Circle(
            radius=0.95, color=C["blue"], stroke_width=3.5,
            fill_color=C["blue"], fill_opacity=0.12,
        )
        croq_circle.move_to(center)
        self.add(croq_circle)

        croq_lbl = Text(
            "Croqtile",
            font_size=18, color=C["blue"], font="Monospace",
        )
        croq_lbl.move_to(croq_circle.get_center() + UP * 0.25)
        self.add(croq_lbl)

        croq_sub = Text(
            "dma.copy \u00b7 tma.copy",
            font_size=13, color=C["fg2"], font="Monospace",
        )
        croq_sub.move_to(croq_circle.get_center() + DOWN * 0.08)
        self.add(croq_sub)

        croq_note = Text(
            "always fast",
            font_size=13, color=C["blue"], font="Monospace",
        )
        croq_note.move_to(croq_circle.get_center() + DOWN * 0.4)
        self.add(croq_note)

        # Bad patterns in the outer ring (between CUDA and sweet spot)
        bad_patterns = [
            ("strided\nuncoalesced",   2.35,  40),
            ("bank\nconflicts",        2.35, 100),
            ("misaligned\nswizzle",    2.35, 170),
            ("scalar\nloads",          2.35, 220),
            ("divergent\naddressing",  2.35, 310),
        ]
        import numpy as np
        for txt, radius, angle_deg in bad_patterns:
            angle = np.deg2rad(angle_deg)
            pos = center + RIGHT * radius * np.cos(angle) + UP * radius * np.sin(angle)
            x_mark = Text("\u2717", font_size=16, color=C["red"], font="Monospace")
            x_mark.move_to(pos)
            label = Text(txt, font_size=10, color=C["red"], font="Monospace")
            label.next_to(x_mark, DOWN, buff=0.06)
            self.add(x_mark, label)

        foot = Text(
            "Croqtile trades expressiveness for guaranteed performance \u2014"
            " every expressible pattern is in the sweet spot",
            font_size=12, color=C["dim"], font="Monospace",
        )
        foot.to_edge(DOWN, buff=0.25)
        self.add(foot)
