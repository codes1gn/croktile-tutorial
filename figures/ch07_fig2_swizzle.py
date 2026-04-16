"""
Ch07 Fig2: Shared-memory bank conflicts without swizzle vs XOR swizzle.
Top: multiple lanes map to the same bank (serialized).
Bottom: XOR-remapped indices hit distinct banks.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from manim import *
from theme import parse_theme
import numpy as np

C, THEME = parse_theme()


class SwizzleBanks(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text(
            "Bank conflicts: no swizzle vs XOR swizzle",
            font_size=28, color=C["fg"], font="Monospace",
        )
        title.to_edge(UP, buff=0.3)
        self.add(title)

        n_banks = 8
        bank_w = 0.65
        bank_h = 0.55
        gap = 0.08

        def draw_bank_row(y_center, conflict_bank=-1):
            x_start = -(n_banks * (bank_w + gap)) / 2 + bank_w / 2
            banks = []
            for b in range(n_banks):
                is_conflict = (b == conflict_bank)
                fill = C["red"] if is_conflict else C["fill"]
                stroke = C["red"] if is_conflict else C["stroke"]
                r = Rectangle(
                    width=bank_w,
                    height=bank_h,
                    fill_color=fill,
                    fill_opacity=0.40 if is_conflict else 0.25,
                    stroke_color=stroke,
                    stroke_width=2 if is_conflict else 1.5,
                )
                r.move_to(np.array([x_start + b * (bank_w + gap), y_center, 0]))
                num = Text(f"B{b}", font_size=13, color=C["fg2"], font="Monospace")
                num.move_to(r.get_center())
                self.add(r, num)
                banks.append(r)
            return banks

        # ---------------- Top panel: without swizzle ----------------
        top_bank_y = 1.05
        top_heading = Text(
            "Without swizzle: lanes collide on bank 0",
            font_size=16, color=C["fg"], font="Monospace",
        )
        top_heading.move_to(UP * 2.8)
        self.add(top_heading)

        banks1 = draw_bank_row(top_bank_y, conflict_bank=0)

        # top_lane_x = [-3, -2, -1, 0]
        top_lane_x = [-4.35, -3.25, -2.10, -0.95]
        end_offsets = [-0.23, -0.08, 0.08, 0.23]
        arc_angles = [-0.26, -0.12, 0.10, 0.24]
        lane_y_top = 2
        for i in range(4):
            dot = Dot(point=np.array([top_lane_x[i], lane_y_top, 0]), radius=0.1, color=C["orange"])
            lbl = Text(f"Lane {i}", font_size=12, color=C["orange"], font="Monospace")
            lbl.next_to(dot, UP, buff=0.08)
            arr = CurvedArrow(
                dot.get_bottom(),
                banks1[0].get_top() + RIGHT * end_offsets[i],
                angle=arc_angles[i],
                color=C["orange"],
                stroke_width=2.5,
                tip_length=0.16,
            )
            self.add(dot, lbl, arr)

        clash = Text(
            "4-way bank conflict  \u2192  serialized (bandwidth / 4)",
            font_size=15,
            color=C["red"],
            font="Monospace",
        )
        clash.move_to(UP * 0.4)
        self.add(clash)

        sep = DashedLine(
            LEFT * 5 + DOWN * 0.15,
            RIGHT * 5 + DOWN * 0.15,
            color=C["dim"],
            stroke_width=1.5,
            dash_length=0.1,
        )
        self.add(sep)

        # ---------------- Bottom panel: XOR swizzle ----------------
        bot_heading = Text(
            "With XOR swizzle: lanes map to distinct banks",
            font_size=16, color=C["fg"], font="Monospace",
        )
        bot_heading.move_to(UP * (-0.67))
        self.add(bot_heading)

        bot_bank_y = -2.3
        banks2 = draw_bank_row(bot_bank_y, conflict_bank=-1)

        targets = [0, 3, 5, 6]
        lane_y_bot = -1.3
        for i, bidx in enumerate(targets):
            tgt = banks2[bidx]
            dot = Dot(point=np.array([top_lane_x[i], lane_y_bot, 0]), radius=0.1, color=C["green"])
            lbl = Text(f"Lane {i}", font_size=12, color=C["green"], font="Monospace")
            lbl.next_to(dot, UP, buff=0.03)
            arr = Arrow(
                dot.get_bottom(),
                tgt.get_top(),
                buff=0.03,
                stroke_width=2.5,
                color=C["green"],
                max_tip_length_to_length_ratio=0.05,
            )
            self.add(dot, lbl, arr)

        ok = Text(
            "conflict-free  \u2192  full bandwidth",
            font_size=15, color=C["green"], font="Monospace",
        )
        ok.move_to(DOWN * 3)
        self.add(ok)

        foot = Text(
            "dma.copy.swiz<N> / tma.copy.swiz<N> + mma.load.swiz<N> must use matching N",
            font_size=13, color=C["fg3"], font="Monospace",
        )
        foot.to_edge(DOWN, buff=0.3)
        self.add(foot)
