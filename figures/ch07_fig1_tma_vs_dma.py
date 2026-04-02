"""
Ch07 Fig1: dma.copy (software-driven cooperative loads) vs tma.copy (descriptor + TMA hardware).
Left: threads compute addresses and participate in the transfer.
Right: one descriptor issues a bulk tensor copy; TMA handles multi-dimensional addressing.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from manim import *

from theme import parse_theme

C, THEME = parse_theme()


class TmaVsDma(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text(
            "Software DMA vs Tensor Memory Accelerator (TMA)",
            font_size=24,
            color=C["fg"],
            font="Monospace",
        )
        title.to_edge(UP, buff=0.35)
        self.add(title)

        sep = DashedLine(
            start=UP * 2.6 + DOWN * 3.2,
            end=DOWN * 2.8,
            color=C["dim"],
            stroke_width=2,
            dash_length=0.12,
        )
        self.add(sep)

        # --- Left: dma.copy ---
        left_title = Text("dma.copy", font_size=20, color=C["orange"], font="Monospace")
        left_title.move_to(LEFT * 3.35 + UP * 2.05)
        sub_l = Text(
            "threads + address math per lane",
            font_size=13,
            color=C["fg2"],
            font="Monospace",
        )
        sub_l.next_to(left_title, DOWN, buff=0.12)
        self.add(left_title, sub_l)

        threads = VGroup()
        for t in range(4):
            tid = Text(f"T{t}", font_size=14, color=C["elem"], font="Monospace")
            addr = Text(f"base+{t}·stride", font_size=12, color=C["fg3"], font="Monospace")
            addr.next_to(tid, RIGHT, buff=0.15)
            row = VGroup(tid, addr)
            row.move_to(LEFT * 3.35 + UP * (1.15 - t * 0.52))
            threads.add(row)
        self.add(threads)

        gmem_l = Rectangle(
            width=3.2,
            height=1.1,
            fill_color=C["global_c"],
            fill_opacity=0.15,
            stroke_color=C["global_c"],
            stroke_width=2,
        )
        gmem_l.move_to(LEFT * 3.35 + DOWN * 0.35)
        gmem_lbl = Text("global tile", font_size=13, color=C["global_c"], font="Monospace")
        gmem_lbl.move_to(gmem_l.get_top() + DOWN * 0.22)
        self.add(gmem_l, gmem_lbl)

        smem_l = Rectangle(
            width=3.2,
            height=1.1,
            fill_color=C["shared_c"],
            fill_opacity=0.15,
            stroke_color=C["shared_c"],
            stroke_width=2,
        )
        smem_l.move_to(LEFT * 3.35 + DOWN * 1.95)
        smem_ll = Text("shared staging", font_size=13, color=C["shared_c"], font="Monospace")
        smem_ll.move_to(smem_l.get_top() + DOWN * 0.22)
        self.add(smem_l, smem_ll)

        for t in range(4):
            a1 = Arrow(
                gmem_l.get_left() + RIGHT * 0.4 + UP * (0.35 - t * 0.22),
                smem_l.get_left() + RIGHT * 0.4 + UP * (0.35 - t * 0.22),
                buff=0.05,
                stroke_width=2,
                color=C["arrow"],
                max_tip_length_to_length_ratio=0.12,
            )
            self.add(a1)

        note_l = Text(
            "warps cooperate; each lane issues loads",
            font_size=13,
            color=C["dim"],
            font="Monospace",
        )
        note_l.move_to(LEFT * 3.35 + DOWN * 2.85)
        self.add(note_l)

        # --- Right: tma.copy ---
        right_title = Text("tma.copy", font_size=20, color=C["blue"], font="Monospace")
        right_title.move_to(RIGHT * 3.35 + UP * 2.05)
        sub_r = Text(
            "descriptor + dedicated hardware path",
            font_size=13,
            color=C["fg2"],
            font="Monospace",
        )
        sub_r.next_to(right_title, DOWN, buff=0.12)
        self.add(right_title, sub_r)

        desc = RoundedRectangle(
            corner_radius=0.08,
            width=2.8,
            height=0.95,
            fill_color=C["fill"],
            fill_opacity=0.9,
            stroke_color=C["blue"],
            stroke_width=2,
        )
        desc.move_to(RIGHT * 3.35 + UP * 0.95)
        desc_txt = Text("tensor descriptor", font_size=14, color=C["blue"], font="Monospace")
        desc_txt.move_to(desc.get_center() + UP * 0.12)
        desc_sub = Text("(layout, tile, origin)", font_size=13, color=C["fg2"], font="Monospace")
        desc_sub.move_to(desc.get_center() + DOWN * 0.18)
        self.add(desc, desc_txt, desc_sub)

        tma_box = RoundedRectangle(
            corner_radius=0.1,
            width=2.6,
            height=1.15,
            fill_color=C["purple"],
            fill_opacity=0.22,
            stroke_color=C["purple"],
            stroke_width=2.5,
        )
        tma_box.move_to(RIGHT * 3.35 + DOWN * 0.35)
        tma_lbl = Text("TMA unit", font_size=16, color=C["purple"], font="Monospace")
        tma_sub = Text("multi-dim addressing in HW", font_size=13, color=C["fg2"], font="Monospace")
        tma_lbl.move_to(tma_box.get_center() + UP * 0.15)
        tma_sub.move_to(tma_box.get_center() + DOWN * 0.2)
        self.add(tma_box, tma_lbl, tma_sub)

        gmem_r = Rectangle(
            width=2.4,
            height=0.55,
            fill_color=C["global_c"],
            fill_opacity=0.2,
            stroke_color=C["global_c"],
            stroke_width=1.5,
        )
        gmem_r.move_to(RIGHT * 3.35 + UP * 2.55)
        gmem_rl = Text("global", font_size=14, color=C["global_c"], font="Monospace")
        gmem_rl.move_to(gmem_r)
        self.add(gmem_r, gmem_rl)

        smem_r = Rectangle(
            width=2.4,
            height=0.55,
            fill_color=C["shared_c"],
            fill_opacity=0.2,
            stroke_color=C["shared_c"],
            stroke_width=1.5,
        )
        smem_r.move_to(RIGHT * 3.35 + DOWN * 1.55)
        smem_rl = Text("shared", font_size=14, color=C["shared_c"], font="Monospace")
        smem_rl.move_to(smem_r)
        self.add(smem_r, smem_rl)

        ar1 = Arrow(
            gmem_r.get_bottom(),
            desc.get_top(),
            buff=0.08,
            stroke_width=3,
            color=C["arrow"],
            max_tip_length_to_length_ratio=0.14,
        )
        ar2 = Arrow(
            desc.get_bottom(),
            tma_box.get_top(),
            buff=0.08,
            stroke_width=3,
            color=C["arrow"],
            max_tip_length_to_length_ratio=0.14,
        )
        ar3 = Arrow(
            tma_box.get_bottom(),
            smem_r.get_top(),
            buff=0.08,
            stroke_width=3,
            color=C["green"],
            max_tip_length_to_length_ratio=0.14,
        )
        self.add(ar1, ar2, ar3)

        note_r = Text(
            "one issue · bulk tile · minimal thread work",
            font_size=13,
            color=C["dim"],
            font="Monospace",
        )
        note_r.move_to(RIGHT * 3.35 + DOWN * 2.35)
        self.add(note_r)

        foot = Text(
            "Hopper (SM90+): TMA replaces cooperative thread loads for tensor ingress",
            font_size=12,
            color=C["fg3"],
            font="Monospace",
        )
        foot.to_edge(DOWN, buff=0.35)
        self.add(foot)
