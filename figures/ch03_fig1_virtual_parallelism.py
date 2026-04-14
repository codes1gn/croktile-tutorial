"""
Figure 1: Virtual Parallelism — abstract tasks scheduled on different hardware.
Shows that "parallel" is a virtual concept; the same 8 tasks can be mapped
to 1-at-a-time (sequential), 4-at-a-time (CPU), or 8-at-a-time (GPU).
Visual key: number of columns = number of time steps. More stacking = more
simultaneous execution.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from theme import parse_theme
from manim import *

C, THEME = parse_theme()

class VirtualParallelism(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text("Parallelism Is a Virtual Concept", font_size=26,
                      color=C["fg"], font="Monospace")
        title.to_edge(UP, buff=0.3)
        self.add(title)

        subtitle = Text("8 independent tasks — same work, different schedules",
                         font_size=14, color=C["fg2"], font="Monospace")
        subtitle.next_to(title, DOWN, buff=0.15)
        self.add(subtitle)

        task_colors = [C["blue"], C["orange"], C["green"], C["purple"],
                       C["teal"], C["pink"], C["yellow"], C["red"]]

        TW = 0.45
        TH = 0.3
        LABEL_X = -5.2
        TASK_X = -2.8

        # --- Sequential: 8 time steps, 1 task each ---
        seq_label = Text("Sequential", font_size=15, color=C["elem"],
                         font="Monospace")
        seq_label.move_to(LEFT * 5.2 + UP * 1.8)
        seq_sub = Text("1 core", font_size=11, color=C["fg3"], font="Monospace")
        seq_sub.next_to(seq_label, DOWN, buff=0.06)
        self.add(seq_label, seq_sub)

        for i in range(8):
            r = Rectangle(width=TW, height=TH, fill_color=task_colors[i],
                           fill_opacity=0.7, stroke_color=task_colors[i], stroke_width=1)
            r.move_to(RIGHT * (TASK_X + i * 0.55) + UP * 1.8)
            t = Text(f"T{i}", font_size=10, color=C["fg"], font="Monospace").move_to(r)
            self.add(r, t)

        arr1 = Arrow(LEFT * 3.0 + UP * 1.3, RIGHT * 2.0 + UP * 1.3, buff=0,
                     stroke_width=1, color=C["dim"], max_tip_length_to_length_ratio=0.03)
        self.add(arr1)
        self.add(Text("time →", font_size=10, color=C["dim"], font="Monospace"
                       ).next_to(arr1, DOWN, buff=0.03).align_to(arr1, RIGHT))

        # --- CPU 4-wide: 2 time steps, 4 tasks stacked each ---
        cpu_label = Text("4-wide (CPU)", font_size=15, color=C["blue"],
                         font="Monospace")
        cpu_label.move_to(LEFT * 5.2 + UP * 0.2)
        cpu_sub = Text("4 cores", font_size=11, color=C["fg3"], font="Monospace")
        cpu_sub.next_to(cpu_label, DOWN, buff=0.06)
        self.add(cpu_label, cpu_sub)

        for step in range(2):
            for lane in range(4):
                idx = step * 4 + lane
                r = Rectangle(width=TW, height=TH, fill_color=task_colors[idx],
                               fill_opacity=0.7, stroke_color=task_colors[idx], stroke_width=1)
                r.move_to(RIGHT * (TASK_X + step * 2.2) + UP * (0.7 - lane * 0.38))
                t = Text(f"T{idx}", font_size=10, color=C["fg"], font="Monospace").move_to(r)
                self.add(r, t)

        arr2 = Arrow(LEFT * 3.0 + DOWN * 1.0, RIGHT * 2.0 + DOWN * 1.0, buff=0,
                     stroke_width=1, color=C["dim"], max_tip_length_to_length_ratio=0.03)
        self.add(arr2)
        self.add(Text("time →", font_size=10, color=C["dim"], font="Monospace"
                       ).next_to(arr2, DOWN, buff=0.03).align_to(arr2, RIGHT))

        # --- GPU 8-wide: 1 time step, all 8 stacked ---
        gpu_label = Text("8-wide (GPU)", font_size=15, color=C["green"],
                         font="Monospace")
        gpu_label.move_to(LEFT * 5.2 + DOWN * 1.9)
        gpu_sub = Text("1 warp", font_size=11, color=C["fg3"], font="Monospace")
        gpu_sub.next_to(gpu_label, DOWN, buff=0.06)
        self.add(gpu_label, gpu_sub)

        gpu_top_y = -1.2
        gpu_spacing = 0.25
        for lane in range(8):
            r = Rectangle(width=TW, height=0.22, fill_color=task_colors[lane],
                           fill_opacity=0.7, stroke_color=task_colors[lane], stroke_width=1)
            y = gpu_top_y - lane * gpu_spacing
            r.move_to(RIGHT * TASK_X + UP * y)
            t = Text(f"T{lane}", font_size=9, color=C["fg"], font="Monospace").move_to(r)
            self.add(r, t)

        gpu_bot_y = gpu_top_y - 7 * gpu_spacing
        brace_line = Line(
            RIGHT * (TASK_X + TW / 2 + 0.1) + UP * gpu_top_y,
            RIGHT * (TASK_X + TW / 2 + 0.1) + UP * gpu_bot_y,
        )
        brace = Brace(brace_line, direction=RIGHT, buff=0.05, color=C["fg2"])
        brace_lbl = Text("1 step", font_size=10, color=C["fg2"],
                          font="Monospace")
        brace_lbl.next_to(brace, RIGHT, buff=0.08)
        self.add(brace, brace_lbl)

        # Key takeaway
        box = Rectangle(width=7.5, height=0.45, fill_color=C["fill"],
                        fill_opacity=0.8, stroke_color=C["green"], stroke_width=1)
        box.move_to(RIGHT * 0.8 + DOWN * 3.5)
        msg = Text("Same 8 tasks. The hardware decides what runs simultaneously.",
                    font_size=11, color=C["fg"], font="Monospace")
        msg.move_to(box)
        self.add(box, msg)
