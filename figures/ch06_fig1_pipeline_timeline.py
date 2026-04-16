"""
Ch06 Fig1: Sequential load-then-compute vs double-buffered pipeline.
Gantt-style timelines showing idle gaps vs overlapping load and compute.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from manim import *

from theme import parse_theme

C, THEME = parse_theme()


class PipelineTimeline(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text(
            "Sequential vs double-buffered (K-tile timeline)",
            font_size=22,
            color=C["fg"],
            font="Monospace",
        )
        title.to_edge(UP, buff=0.3)
        self.add(title)

        # --- Legend ---
        leg_load = Rectangle(
            width=0.35,
            height=0.22,
            fill_color=C["blue"],
            fill_opacity=0.55,
            stroke_color=C["blue"],
            stroke_width=1,
        )
        leg_comp = Rectangle(
            width=0.35,
            height=0.22,
            fill_color=C["orange"],
            fill_opacity=0.55,
            stroke_color=C["orange"],
            stroke_width=1,
        )
        t_load = Text("Load (DMA)", font_size=14, color=C["fg2"], font="Monospace")
        t_comp = Text("Compute (MMA)", font_size=14, color=C["fg2"], font="Monospace")
        leg_row = VGroup(
            VGroup(leg_load, t_load).arrange(RIGHT, buff=0.15),
            VGroup(leg_comp, t_comp).arrange(RIGHT, buff=0.15),
        ).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        leg_row.to_edge(UP, buff=0.85).to_edge(RIGHT, buff=0.4)
        self.add(leg_row)

        # Time axis helper
        def bar(x0, w, y, color, label=None, h=0.38):
            r = Rectangle(
                width=w,
                height=h,
                fill_color=color,
                fill_opacity=0.55,
                stroke_color=color,
                stroke_width=1.2,
            )
            r.move_to(np.array([x0 + w / 2, y, 0]))
            g = VGroup(r)
            if label:
                t = Text(label, font_size=12, color=C["bg"] if THEME == "light" else C["fg"], font="Monospace")
                t.scale(min(1.0, w / (len(label) * 0.09 + 0.01)))
                t.move_to(r)
                g.add(t)
            return g

        # --- Sequential: one buffer — load and compute cannot overlap across tiles ---
        y_seq = 0.85
        lab_seq = Text(
            "One buffer: load and compute alternate (staircase)",
            font_size=16,
            color=C["purple"],
            font="Monospace",
        )
        lab_seq.move_to(UP * y_seq + LEFT * 2.8)
        self.add(lab_seq)

        axis_y = y_seq - 0.95
        time_lbl = Text("time →", font_size=12, color=C["dim"], font="Monospace")
        wL, wC = 1.0, 1.0
        gap = 0.05
        x0 = -4.2

        # Idle tint regions (sequential: other unit idle)
        idle1 = Rectangle(
            width=1.15,
            height=0.95,
            fill_color=C["red"],
            fill_opacity=0.08,
            stroke_width=0,
        )
        # Align idle windows with C1/C2 slots in the sequential MMA row.
        c1_center_x = x0 + wL + wC + gap + wL + gap + wC / 2
        idle1.move_to(np.array([c1_center_x, axis_y - 0.05, 0]))
        idle2 = Rectangle(
            width=1.15,
            height=0.95,
            fill_color=C["red"],
            fill_opacity=0.08,
            stroke_width=0,
        )
        c2_center_x = x0 + 2 * (wL + wC + gap) + wL + gap + wC / 2
        idle2.move_to(np.array([c2_center_x, axis_y - 0.05, 0]))
        idle_note1 = Text("idle", font_size=12, color=C["dim"], font="Monospace")
        idle_note1.next_to(idle1, UP, buff=0.08)
        idle_note2 = Text("idle", font_size=12, color=C["dim"], font="Monospace")
        idle_note2.next_to(idle2, UP, buff=0.08)
        self.add(idle1, idle2, idle_note1, idle_note2)

        seq_dma = VGroup(
            bar(x0, wL, axis_y + 0.35, C["blue"], "L0"),
            bar(x0 + wL + wC + gap, wL, axis_y + 0.35, C["blue"], "L1"),
            bar(x0 + 2 * (wL + wC + gap), wL, axis_y + 0.35, C["blue"], "L2"),
        )
        seq_mma = VGroup(
            bar(x0 + wL + gap, wC, axis_y - 0.35, C["orange"], "C0"),
            bar(x0 + wL + wC + gap + wL + gap, wC, axis_y - 0.35, C["orange"], "C1"),
            bar(x0 + 2 * (wL + wC + gap) + wL + gap, wC, axis_y - 0.35, C["orange"], "C2"),
        )
        row_dma = Text("DMA", font_size=12, color=C["blue"], font="Monospace").move_to(
            np.array([-5.5, axis_y + 0.35, 0])
        )
        row_mma = Text("MMA", font_size=12, color=C["orange"], font="Monospace").move_to(
            np.array([-5.5, axis_y - 0.35, 0])
        )
        self.add(row_dma, row_mma, seq_dma, seq_mma)

        # --- Double-buffered ---
        y_db = -1.15
        lab_db = Text(
            "Two buffers: overlap next load with current compute",
            font_size=16,
            color=C["green"],
            font="Monospace",
        )
        lab_db.move_to(UP * y_db + LEFT * 2.5)
        self.add(lab_db)

        axis_y2 = y_db - 0.95
        # Keep the time axis hint below both rows to avoid label collisions.
        time_lbl.move_to(np.array([-5.45, axis_y2 - 0.95, 0]))
        self.add(time_lbl)
        x1 = -4.2
        # Pipeline: L0, then L1 overlaps C0, etc.
        db_dma = VGroup(
            bar(x1, wL, axis_y2 + 0.35, C["blue"], "L0"),
            bar(x1 + wL, wL, axis_y2 + 0.35, C["blue"], "L1"),
            bar(x1 + 2 * wL, wL, axis_y2 + 0.35, C["blue"], "L2"),
            bar(x1 + 3 * wL, wL, axis_y2 + 0.35, C["blue"], "L3"),
        )
        db_mma = VGroup(
            bar(x1 + wL, wC, axis_y2 - 0.35, C["orange"], "C0"),
            bar(x1 + wL + wC, wC, axis_y2 - 0.35, C["orange"], "C1"),
            bar(x1 + wL + 2 * wC, wC, axis_y2 - 0.35, C["orange"], "C2"),
        )
        row_dma2 = Text("DMA", font_size=12, color=C["blue"], font="Monospace").move_to(
            np.array([-5.5, axis_y2 + 0.35, 0])
        )
        row_mma2 = Text("MMA", font_size=12, color=C["orange"], font="Monospace").move_to(
            np.array([-5.5, axis_y2 - 0.35, 0])
        )
        self.add(row_dma2, row_mma2, db_dma, db_mma)

        saved = Text(
            "← more useful overlap, less wasted idle",
            font_size=13,
            color=C["fg3"],
            font="Monospace",
        )
        saved.move_to(np.array([3.0, axis_y2, 0]))
        self.add(saved)

        br = Text(
            "(Bar widths are schematic — real kernels depend on tile sizes and latency.)",
            font_size=12,
            color=C["dim"],
            font="Monospace",
        )
        br.to_edge(DOWN, buff=0.35)
        self.add(br)
