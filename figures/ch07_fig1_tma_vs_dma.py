"""
Ch07 Fig1: dma.copy (software-driven cooperative loads) vs tma.copy (descriptor + TMA hardware).
Left: threads compute addresses and participate in the transfer.
Right: one descriptor issues a bulk tensor copy; TMA handles multi-dimensional addressing.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from manim import *
from theme import parse_theme

C, THEME = parse_theme()


class TmaVsDma(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text(
            "Software DMA vs Tensor Memory Accelerator (TMA)",
            font_size=24, color=C["fg"], font="Monospace",
        )
        title.to_edge(UP, buff=0.3)
        self.add(title)

        sep = DashedLine(
            start=UP * 2.3, end=DOWN * 2.8,
            color=C["dim"], stroke_width=2, dash_length=0.12,
        )
        self.add(sep)

        # ═══ Left: dma.copy ═══
        lx = -3.5

        left_title = Text("dma.copy", font_size=20, color=C["orange"], font="Monospace")
        left_title.move_to(RIGHT * lx + UP * 2.0)
        sub_l = Text(
            "threads + address math per lane",
            font_size=13, color=C["fg2"], font="Monospace",
        )
        sub_l.next_to(left_title, DOWN, buff=0.08)
        self.add(left_title, sub_l)

        # Global memory block (top)
        gmem_l = Rectangle(
            width=3.4, height=0.65,
            fill_color=C["global_c"], fill_opacity=0.15,
            stroke_color=C["global_c"], stroke_width=2,
        )
        gmem_l.move_to(RIGHT * lx + UP * 1.1)
        gmem_lbl = Text("global memory", font_size=14, color=C["global_c"], font="Monospace")
        gmem_lbl.move_to(gmem_l)
        self.add(gmem_l, gmem_lbl)

        # Thread lanes (middle): place T0..T3 horizontally.
        lane_y = 0.05
        lane_centers = []
        for t in range(4):
            x = lx - 1.1 + t * 0.75
            center = np.array([x, lane_y, 0])
            lane_centers.append(center)
            lane_box = RoundedRectangle(
                corner_radius=0.05, width=0.52, height=0.28,
                fill_color=C["fill"], fill_opacity=0.85,
                stroke_color=C["elem"], stroke_width=1.5,
            )
            lane_box.move_to(center)
            lane_lbl = Text(f"T{t}", font_size=13, color=C["elem"], font="Monospace")
            lane_lbl.move_to(lane_box)
            self.add(lane_box, lane_lbl)

        # Arrows: global -> lane
        for t in range(4):
            a = Arrow(
                gmem_l.get_bottom() + RIGHT * (-1.0 + t * 0.68),
                lane_centers[t] + UP * 0.2,
                buff=0.04, stroke_width=1.9, color=C["arrow"],
                max_tip_length_to_length_ratio=0.2,
            )
            self.add(a)

        # Shared memory block (bottom)
        smem_l = Rectangle(
            width=3.4, height=0.65,
            fill_color=C["shared_c"], fill_opacity=0.15,
            stroke_color=C["shared_c"], stroke_width=2,
        )
        smem_l.move_to(RIGHT * lx + DOWN * 1.5)
        smem_ll = Text("shared memory", font_size=14, color=C["shared_c"], font="Monospace")
        smem_ll.move_to(smem_l)
        self.add(smem_l, smem_ll)

        # Arrows: lane -> shared
        for t in range(4):
            a = Arrow(
                lane_centers[t] + DOWN * 0.2,
                smem_l.get_top() + RIGHT * (-1.0 + t * 0.68),
                buff=0.04, stroke_width=1.9, color=C["arrow"],
                max_tip_length_to_length_ratio=0.12,
            )
            self.add(a)

        # Offset explanation placed separately under the lane diagram.
        offset_title = Text("per-lane offsets:", font_size=12, color=C["fg2"], font="Monospace")
        offset_title.move_to(RIGHT * lx + DOWN * 1.95)
        offsets_l = Text("T0: base+0\u00b7stride   T1: base+1\u00b7stride",
                         font_size=11, color=C["fg3"], font="Monospace")
        offsets_r = Text("T2: base+2\u00b7stride   T3: base+3\u00b7stride",
                         font_size=11, color=C["fg3"], font="Monospace")
        offsets_l.next_to(offset_title, DOWN, buff=0.08)
        offsets_r.next_to(offsets_l, DOWN, buff=0.06)
        self.add(offset_title, offsets_l, offsets_r)

        note_l = Text(
            "warps cooperate; each lane issues loads",
            font_size=13, color=C["dim"], font="Monospace",
        )
        note_l.move_to(RIGHT * lx + DOWN * 2.6)
        self.add(note_l)

        # ═══ Right: tma.copy ═══
        rx = 3.5

        right_title = Text("tma.copy", font_size=20, color=C["blue"], font="Monospace")
        right_title.move_to(RIGHT * rx + UP * 2.0)
        sub_r = Text(
            "descriptor + dedicated HW engine",
            font_size=13, color=C["fg2"], font="Monospace",
        )
        sub_r.next_to(right_title, DOWN, buff=0.08)
        self.add(right_title, sub_r)

        # Global memory block
        gmem_r = Rectangle(
            width=2.8, height=0.55,
            fill_color=C["global_c"], fill_opacity=0.2,
            stroke_color=C["global_c"], stroke_width=1.5,
        )
        gmem_r.move_to(RIGHT * rx + UP * 1.1)
        gmem_rl = Text("global", font_size=14, color=C["global_c"], font="Monospace")
        gmem_rl.move_to(gmem_r)
        self.add(gmem_r, gmem_rl)

        # Descriptor box
        desc = RoundedRectangle(
            corner_radius=0.08, width=2.8, height=0.70,
            fill_color=C["fill"], fill_opacity=0.9,
            stroke_color=C["blue"], stroke_width=2,
        )
        desc.move_to(RIGHT * rx + UP * 0.1)
        desc_txt = Text("tensor descriptor", font_size=14, color=C["blue"], font="Monospace")
        desc_txt.move_to(desc.get_center() + UP * 0.12)
        desc_sub = Text("(layout, tile, origin)", font_size=13, color=C["fg2"], font="Monospace")
        desc_sub.move_to(desc.get_center() + DOWN * 0.16)
        self.add(desc, desc_txt, desc_sub)

        # TMA unit box
        tma_box = RoundedRectangle(
            corner_radius=0.1, width=2.8, height=0.75,
            fill_color=C["purple"], fill_opacity=0.22,
            stroke_color=C["purple"], stroke_width=2.5,
        )
        tma_box.move_to(RIGHT * rx + DOWN * 1)
        tma_lbl = Text("TMA unit", font_size=16, color=C["purple"], font="Monospace")
        tma_sub = Text("multi-dim addressing in HW", font_size=13, color=C["fg2"], font="Monospace")
        tma_lbl.move_to(tma_box.get_center() + UP * 0.12)
        tma_sub.move_to(tma_box.get_center() + DOWN * 0.16)
        self.add(tma_box, tma_lbl, tma_sub)

        # Shared memory block
        smem_r = Rectangle(
            width=2.8, height=0.55,
            fill_color=C["shared_c"], fill_opacity=0.2,
            stroke_color=C["shared_c"], stroke_width=1.5,
        )
        smem_r.move_to(RIGHT * rx + DOWN * 2)
        smem_rl = Text("shared", font_size=14, color=C["shared_c"], font="Monospace")
        smem_rl.move_to(smem_r)
        self.add(smem_r, smem_rl)

        # Arrows: global -> descriptor -> TMA -> shared
        ar1 = Arrow(
            gmem_r.get_bottom(), desc.get_top(),
            buff=0.06, stroke_width=3, color=C["arrow"],
            max_tip_length_to_length_ratio=0.4,
        )
        ar2 = Arrow(
            desc.get_bottom(), tma_box.get_top(),
            buff=0.06, stroke_width=3, color=C["arrow"],
            max_tip_length_to_length_ratio=0.4,
        )
        ar3 = Arrow(
            tma_box.get_bottom(), smem_r.get_top(),
            buff=0.06, stroke_width=3, color=C["arrow"],
            max_tip_length_to_length_ratio=0.4,
        )
        self.add(ar1, ar2, ar3)

        note_r = Text(
            "one issue \u00b7 bulk tile \u00b7 minimal thread work",
            font_size=13, color=C["dim"], font="Monospace",
        )
        note_r.move_to(RIGHT * rx + DOWN * 2.5)
        self.add(note_r)

        foot = Text(
            "same data-path bandwidth \u2014 TMA advantage is async overlap via dedicated engine",
            font_size=12, color=C["fg3"], font="Monospace",
        )
        foot.to_edge(DOWN, buff=0.3)
        self.add(foot)
