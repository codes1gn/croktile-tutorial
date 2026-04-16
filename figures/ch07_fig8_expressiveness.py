"""
Ch07 Fig8: Expressiveness vs Performance.
Three concentric circles:
  Outer (largest): CUDA expressible range — wide but includes many slow patterns.
  Middle: Croqtile range — restricted but fully contains the sweet spot.
  Inner (smallest): Performance sweet spot — coalesced, conflict-free, aligned.

Key relationship: Croqtile ⊃ sweet spot, CUDA ⊃ Croqtile.
By restricting expressiveness, Croqtile guarantees every program is in the sweet spot.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from theme import parse_theme
from manim import *
import numpy as np

C, THEME = parse_theme()


class ExpressivenessVsPerformance(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text(
            "Expressiveness vs Performance",
            font_size=28, color=C["fg"], font="Monospace",
        )
        title.to_edge(UP, buff=0.25)
        self.add(title)

        sub = Text(
            "data-movement patterns: who can express what?",
            font_size=14, color=C["fg2"], font="Monospace",
        )
        sub.next_to(title, DOWN*1.3, buff=0.08)
        self.add(sub)

        center = DOWN * 0.15

        # Outer: CUDA (largest)
        cuda_r = 2.9
        cuda_circle = Circle(
            radius=cuda_r, color=C["orange"], stroke_width=3,
            fill_color=C["orange"], fill_opacity=0.04,
        )
        cuda_circle.move_to(center)
        self.add(cuda_circle)

        cuda_lbl = Text(
            "CUDA expressible range",
            font_size=16, color=C["orange"], font="Monospace",
        )
        cuda_lbl.move_to(cuda_circle.get_top() + DOWN * 0.8)
        self.add(cuda_lbl)

        # Middle: Croqtile — larger than sweet spot, fully contains it
        croq_r = 1.7
        croq_circle = Circle(
            radius=croq_r, color=C["blue"], stroke_width=3.5,
            fill_color=C["blue"], fill_opacity=0.10,
        )
        croq_circle.move_to(center)
        self.add(croq_circle)

        croq_lbl = Text(
            "Croqtile range",
            font_size=17, color=C["blue"], font="Monospace",
        )
        croq_lbl.move_to(center + UP * 1.10)
        self.add(croq_lbl)

        # Split into 3 parts and distribute evenly along the blue circle.
        croq_items = ["dma.copy", "tma.copy", "swizzle"]
        item_radius = croq_r - 0.45
        item_angles = [165, 90, 15]
        for txt, ang in zip(croq_items, item_angles):
            a = np.deg2rad(ang)
            pos = center + RIGHT * item_radius * np.cos(a) + DOWN * item_radius * np.sin(a)
            t = Text(txt, font_size=12, color=C["blue"], font="Monospace")
            t.move_to(pos)
            self.add(t)

        # Inner: Performance sweet spot (smallest)
        sweet_r = 0.85
        sweet_circle = Circle(
            radius=sweet_r, color=C["green"], stroke_width=3,
            fill_color=C["green"], fill_opacity=0.14,
        )
        sweet_circle.move_to(center)
        self.add(sweet_circle)

        sweet_lbl = Text(
            "Performance",
            font_size=16, color=C["green"], font="Monospace",
        )
        sweet_lbl2 = Text(
            "sweet spot",
            font_size=16, color=C["green"], font="Monospace",
        )
        # Keep the original wording but split into two lines so it can
        # reliably fit inside the sweet-spot circle.
        sweet_sub1 = Text(
            "coalesced  aligned",
            font_size=11, color=C["fg3"], font="Monospace",
        )
        sweet_sub2 = Text(
            " conflict-free",
            font_size=11, color=C["fg3"], font="Monospace",
        )
        sweet_group = VGroup(sweet_lbl, sweet_lbl2, sweet_sub1, sweet_sub2).arrange(
            DOWN, buff=0.07
        )
        sweet_group.move_to(center + DOWN * 0.10)
        # Guarantee text stays inside the green circle even after font/theme changes.
        sweet_group.scale_to_fit_width(sweet_r * 1.6)
        self.add(sweet_group)

        # Bad patterns in the outer ring (CUDA-only zone, outside Croqtile)
        bad_patterns = [
            # (label, radius, angle, label_dx, label_dy)
            ("strided\nuncoalesced",   2.42,  52,  0.15, -0.30),
            ("bank\nconflicts",        2.40, 118, -0.5, -0.50),
            ("misaligned\nswizzle",    2.38, 182, 0.1, -0.10),
            ("scalar\nloads",          2.40, 242,  0.05, -0.0),
            ("divergent\naddressing",  2.44, 322,  0.01, 0.2),
        ]
        for txt, radius, angle_deg, dx, dy in bad_patterns:
            angle = np.deg2rad(angle_deg)
            pos = center + RIGHT * radius * np.cos(angle) + UP * radius * np.sin(angle)
            label = Text(txt, font_size=12, color=C["red"], font="Monospace")
            label.move_to(pos + RIGHT * dx + UP * dy)
            x_mark = Text("\u2717", font_size=18, color=C["red"], font="Monospace")
            x_mark.move_to(label.get_top() + UP * 0.08)
            self.add(x_mark, label)

        # Relationship annotation (outside circles for readability).
        brace_note = Text(
            "CUDA \u2283 Croqtile \u2283 sweet spot",
            font_size=14, color=C["fg2"], font="Monospace",
        )
        brace_note.move_to(center + DOWN * 3.2)
        self.add(brace_note)

        foot = Text(
            "every Croqtile program is in the sweet spot \u2014 trade expressiveness for guaranteed performance",
            font_size=12, color=C["dim"], font="Monospace",
        )
        foot.to_edge(DOWN, buff=0.22)
        self.add(foot)
