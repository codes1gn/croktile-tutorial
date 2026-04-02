"""
Figure 3: The 4-step MMA syntax — fill, load, multiply, store.
Emphasizes these are abstract operations that map to different hardware.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from manim import *
from theme import parse_theme

C, THEME = parse_theme()


class MMASyntax(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text(
            "Croktile's Four-Step MMA Syntax",
            font_size=22, color=C["fg"], font="Monospace",
        )
        title.to_edge(UP, buff=0.3)
        self.add(title)

        steps = [
            ("1. fill", "mma.fill 0.0", "zero accumulator mc", C["yellow"]),
            ("2. load", "mma.load", "smem → ma, mb registers", C["blue"]),
            ("3. multiply", "mma.row.row mc, ma, mb", "C += A × B", C["green"]),
            ("4. store", "mma.store mc, dst", "registers → smem", C["orange"]),
        ]

        y_start = 1.0
        y_step = 1.15
        box_w = 8.0
        box_h = 0.85

        for i, (step, syntax, desc, accent) in enumerate(steps):
            y = y_start - i * y_step
            box = RoundedRectangle(
                corner_radius=0.1, width=box_w, height=box_h,
                fill_color=accent, fill_opacity=0.08,
                stroke_color=accent, stroke_width=1.8,
            )
            box.move_to([0, y, 0])

            step_t = Text(step, font_size=14, color=accent, font="Monospace")
            step_t.move_to(box.get_left() + RIGHT * 1.0)

            syn_t = Text(syntax, font_size=13, color=C["fg"], font="Monospace")
            syn_t.move_to(box.get_center() + LEFT * 0.3)

            desc_t = Text(desc, font_size=11, color=C["dim"], font="Monospace")
            desc_t.move_to(box.get_right() + LEFT * 1.5)

            self.add(box, step_t, syn_t, desc_t)

            if i < len(steps) - 1:
                arr = Arrow(
                    [0, y - box_h/2, 0], [0, y - y_step + box_h/2, 0],
                    buff=0.05, stroke_width=2, color=C["arrow_c"],
                    max_tip_length_to_length_ratio=0.2,
                )
                self.add(arr)

        loop_label = Text(
            "loop 2–3 over K slices; run 4 once",
            font_size=11, color=C["fg3"], font="Monospace",
        )
        brace = BraceBetweenPoints(
            [box_w/2 + 0.3, y_start - y_step + box_h/2, 0],
            [box_w/2 + 0.3, y_start - 2*y_step - box_h/2, 0],
            direction=RIGHT,
            color=C["dim"],
        )
        loop_label.next_to(brace, RIGHT, buff=0.1)
        self.add(brace, loop_label)

        foot = Text(
            "Abstract operations — not tied to GPU tensor cores; any 2D contraction hardware can map here.",
            font_size=10, color=C["fg3"], font="Monospace",
        )
        foot.to_edge(DOWN, buff=0.3)
        self.add(foot)
