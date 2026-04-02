"""
Ch07 Fig2: Shared-memory bank conflicts without swizzle vs XOR swizzle.
Top: multiple lanes map to the same bank (serialized).
Bottom: XOR-remapped indices hit distinct banks.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from manim import *
from theme import parse_theme

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

        def make_bank_strip(y_center, heading, conflict_bank=-1):
            grp = VGroup()
            lbl = Text(heading, font_size=16, color=C["fg"], font="Monospace")
            lbl.move_to(UP * (y_center + bank_h / 2 + 0.35))
            grp.add(lbl)

            x_start = -(n_banks * (bank_w + gap)) / 2 + bank_w / 2
            bank_rects = []
            for b in range(n_banks):
                is_conflict = (b == conflict_bank)
                fill = C["red"] if is_conflict else C["fill"]
                stroke = C["red"] if is_conflict else C["stroke"]
                opacity = 0.4 if is_conflict else 0.25
                r = Rectangle(
                    width=bank_w, height=bank_h,
                    fill_color=fill, fill_opacity=opacity,
                    stroke_color=stroke, stroke_width=2 if is_conflict else 1.5,
                )
                r.move_to(RIGHT * (x_start + b * (bank_w + gap)) + UP * y_center)
                num = Text(f"B{b}", font_size=13, color=C["fg2"], font="Monospace")
                num.move_to(r.get_center())
                grp.add(r, num)
                bank_rects.append(r)
            return grp, bank_rects

        # --- Top: conflict case ---
        strip1, banks1 = make_bank_strip(1.1, "Without swizzle: lanes collide on bank 0", conflict_bank=0)
        self.add(strip1)

        lane_y = 1.1 + bank_h / 2 + 0.75
        for i in range(4):
            dot = Dot(
                point=LEFT * 1.5 + RIGHT * i * 0.6 + UP * lane_y,
                radius=0.1, color=C["orange"],
            )
            lbl = Text(f"Lane {i}", font_size=12, color=C["orange"], font="Monospace")
            lbl.next_to(dot, UP, buff=0.08)
            arr = Arrow(
                dot.get_center(),
                banks1[0].get_top() + RIGHT * (i * 0.12 - 0.18),
                buff=0.08, stroke_width=2.5, color=C["orange"],
                max_tip_length_to_length_ratio=0.18,
            )
            self.add(dot, lbl, arr)

        clash = Text(
            "4-way bank conflict  \u2192  serialized (bandwidth / 4)",
            font_size=15, color=C["red"], font="Monospace",
        )
        clash.move_to(UP * 0.25)
        self.add(clash)

        sep = DashedLine(
            LEFT * 5 + DOWN * 0.15, RIGHT * 5 + DOWN * 0.15,
            color=C["dim"], stroke_width=1.5, dash_length=0.1,
        )
        self.add(sep)

        # --- Bottom: swizzled case ---
        strip2, banks2 = make_bank_strip(-1.3, "With XOR swizzle: lanes map to distinct banks", conflict_bank=-1)
        self.add(strip2)

        lane_y2 = -1.3 + bank_h / 2 + 0.75
        targets = [0, 3, 5, 6]
        for i in range(4):
            dot = Dot(
                point=LEFT * 1.5 + RIGHT * i * 0.6 + UP * lane_y2,
                radius=0.1, color=C["green"],
            )
            lbl = Text(f"Lane {i}", font_size=12, color=C["green"], font="Monospace")
            lbl.next_to(dot, UP, buff=0.08)
            tgt = banks2[targets[i]]
            arr = Arrow(
                dot.get_center(), tgt.get_top(),
                buff=0.08, stroke_width=2.5, color=C["green"],
                max_tip_length_to_length_ratio=0.16,
            )
            self.add(dot, lbl, arr)

        ok = Text(
            "conflict-free  \u2192  full bandwidth",
            font_size=15, color=C["green"], font="Monospace",
        )
        ok.move_to(DOWN * 2.25)
        self.add(ok)

        foot = Text(
            "dma.copy.swiz<N> / tma.copy.swiz<N> + mma.load.swiz<N> must use matching N",
            font_size=13, color=C["fg3"], font="Monospace",
        )
        foot.to_edge(DOWN, buff=0.3)
        self.add(foot)
