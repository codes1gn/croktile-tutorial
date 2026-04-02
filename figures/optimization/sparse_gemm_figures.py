"""
Part II — Sparse GEMM figures.
Generates SVG diagrams for the 2:4 structured sparsity optimization story.
Run: manim -qh --format=svg sparse_gemm_figures.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from theme import parse_theme
from manim import *

C, THEME = parse_theme()


class SparsityPattern(Scene):
    """Fig 1: 2:4 structured sparsity — showing the pattern and metadata."""

    def construct(self):
        self.camera.background_color = C["bg"]
        title = Text("2:4 Structured Sparsity: Pattern and Metadata", font_size=26, weight=BOLD)
        title.to_edge(UP, buff=0.4)
        self.add(title)

        dense_label = Text("Dense (original)", font_size=16, color=C["fg3"])
        dense_label.move_to(LEFT * 4 + UP * 1.5)
        self.add(dense_label)

        dense_grid = VGroup()
        dense_vals = [3.1, 0.0, 2.7, 0.0, 1.5, 0.0, 0.8, 4.2,
                      0.0, 1.9, 0.0, 3.3, 2.1, 0.0, 0.0, 5.1]
        for i in range(16):
            row, col = i // 8, i % 8
            is_nonzero = dense_vals[i] != 0.0
            r = Square(
                side_length=0.45,
                fill_color=C["blue_tile"] if is_nonzero else C["gray_bg"],
                fill_opacity=0.8 if is_nonzero else 0.3,
                stroke_width=1
            )
            val_text = Text(f"{dense_vals[i]:.1f}" if is_nonzero else "0", font_size=8,
                            color=C["fg"] if is_nonzero else C["fg3"])
            val_text.move_to(r)
            r.move_to(LEFT * 4 + RIGHT * col * 0.5 + UP * (0.5 - row * 0.5))
            val_text.move_to(r)
            dense_grid.add(VGroup(r, val_text))
        self.add(dense_grid)

        group_braces = VGroup()
        for g in range(4):
            base_col = g * 2
            b = Brace(
                VGroup(dense_grid[base_col][0], dense_grid[base_col + 1][0]),
                DOWN, buff=0.05
            )
            bt = Text("grp", font_size=8, color=C["fg3"])
            bt.next_to(b, DOWN, buff=0.03)
            group_braces.add(VGroup(b, bt))
        for g in range(4):
            base_col = 8 + g * 2
            b = Brace(
                VGroup(dense_grid[base_col][0], dense_grid[base_col + 1][0]),
                DOWN, buff=0.05
            )
            bt = Text("grp", font_size=8, color=C["fg3"])
            bt.next_to(b, DOWN, buff=0.03)
            group_braces.add(VGroup(b, bt))
        self.add(group_braces)

        arrow_mid = Arrow(LEFT * 0.3, RIGHT * 0.8, color=C["orange_tile"], stroke_width=3)
        arrow_label = Text("2:4\ncompress", font_size=12, color=C["orange_tile"])
        arrow_label.next_to(arrow_mid, UP, buff=0.1)
        arrow_mid.move_to(RIGHT * 0.5 + UP * 0.25)
        arrow_label.move_to(RIGHT * 0.5 + UP * 0.9)
        self.add(arrow_mid, arrow_label)

        packed_label = Text("Packed (2:4)", font_size=16, color=C["blue_tile"])
        packed_label.move_to(RIGHT * 3.5 + UP * 1.5)
        self.add(packed_label)

        packed_vals = [3.1, 2.7, 1.5, 4.2, 1.9, 3.3, 2.1, 5.1]
        packed_grid = VGroup()
        for i, v in enumerate(packed_vals):
            row, col = i // 4, i % 4
            r = Square(
                side_length=0.55,
                fill_color=C["blue_tile"], fill_opacity=0.8,
                stroke_width=1
            )
            t = Text(f"{v:.1f}", font_size=9, color=C["fg"])
            r.move_to(RIGHT * 3.5 + RIGHT * col * 0.6 + UP * (0.3 - row * 0.6) + LEFT * 0.9)
            t.move_to(r)
            packed_grid.add(VGroup(r, t))
        self.add(packed_grid)

        meta_label = Text("Metadata (indices)", font_size=14, color=C["teal"], weight=BOLD)
        meta_label.move_to(RIGHT * 3.5 + DOWN * 1.3)
        self.add(meta_label)

        meta_vals = ["0,2", "0,3", "0,2", "1,3"]
        meta_grid = VGroup()
        for i, mv in enumerate(meta_vals):
            r = RoundedRectangle(
                width=0.8, height=0.4, corner_radius=0.05,
                fill_color=C["teal"], fill_opacity=0.3,
                stroke_color=C["teal"], stroke_width=1
            )
            t = Text(mv, font_size=10, color=C["teal"])
            r.move_to(RIGHT * 2.1 + RIGHT * i * 0.9 + DOWN * 1.8)
            t.move_to(r)
            meta_grid.add(VGroup(r, t))
        self.add(meta_grid)

        meta_note = Text(
            "Metadata tells hardware which 2 of 4 elements are nonzero per group",
            font_size=13, color=C["fg3"], slant=ITALIC
        )
        meta_note.to_edge(DOWN, buff=0.5)
        self.add(meta_note)

        cost_note = Text(
            "Tradeoff: 2× operand compression, but metadata is extra traffic on every K tile",
            font_size=13, color=C["red_accent"]
        )
        cost_note.to_edge(DOWN, buff=0.2)
        self.add(cost_note)


class MetadataBottleneck(Scene):
    """Fig 2: Metadata as the bottleneck — scalar loads vs TMA-staged."""

    def construct(self):
        self.camera.background_color = C["bg"]
        title = Text("Metadata Delivery: Scalar vs TMA-Staged", font_size=26, weight=BOLD)
        title.to_edge(UP, buff=0.4)
        self.add(title)

        left_title = Text("Before: Scalar metadata loads", font_size=16, color=C["red_accent"], weight=BOLD)
        left_title.move_to(LEFT * 3.5 + UP * 1.5)
        self.add(left_title)

        def make_k_loop(x_center, meta_style, meta_color):
            blocks = VGroup()
            labels = ["TMA\noperands", meta_style, "WGMMA", "TMA\noperands", meta_style, "WGMMA"]
            colors = [C["orange_tile"], meta_color, C["purple_role"], C["orange_tile"], meta_color, C["purple_role"]]
            widths = [1.2, 0.8, 1.2, 1.2, 0.8, 1.2]
            for i, (lbl, clr, w) in enumerate(zip(labels, colors, widths)):
                r = Rectangle(
                    width=w, height=0.5,
                    fill_color=clr, fill_opacity=0.7,
                    stroke_width=1
                )
                t = Text(lbl, font_size=9, color=C["fg"])
                t.move_to(r)
                blocks.add(VGroup(r, t))
            blocks.arrange(DOWN, buff=0.06)
            blocks.move_to(RIGHT * x_center)
            return blocks

        left_loop = make_k_loop(-3.5, "Scalar\nmeta ⚠", C["red_accent"])
        self.add(left_loop)

        right_title = Text("After: TMA-staged metadata", font_size=16, color=C["green_ok"], weight=BOLD)
        right_title.move_to(RIGHT * 3.5 + UP * 1.5)
        self.add(right_title)

        right_blocks = VGroup()
        r_labels = ["TMA ops\n+ meta", "WGMMA", "TMA ops\n+ meta", "WGMMA"]
        r_colors = [C["orange_tile"], C["purple_role"], C["orange_tile"], C["purple_role"]]
        for lbl, clr in zip(r_labels, r_colors):
            r = Rectangle(
                width=1.4, height=0.5,
                fill_color=clr, fill_opacity=0.7,
                stroke_width=1
            )
            t = Text(lbl, font_size=9, color=C["fg"])
            t.move_to(r)
            right_blocks.add(VGroup(r, t))
        right_blocks.arrange(DOWN, buff=0.06)
        right_blocks.move_to(RIGHT * 3.5 + DOWN * 0.2)
        self.add(right_blocks)

        tma_badge = RoundedRectangle(
            width=1.8, height=0.35, corner_radius=0.08,
            fill_color=C["teal"], fill_opacity=0.4,
            stroke_color=C["teal"]
        )
        tma_badge_t = Text("meta on TMA plane ✓", font_size=10, color=C["teal"])
        tma_badge_t.move_to(tma_badge)
        tma_badge_g = VGroup(tma_badge, tma_badge_t)
        tma_badge_g.next_to(right_blocks, RIGHT, buff=0.3)
        self.add(tma_badge_g)

        comparison = VGroup()
        comp_data = [
            ("FP16 baseline", "368", "Scalar meta"),
            ("E4M3 iter001", "759", "TMA meta (+13%)"),
            ("FP16 iter143", "655", "TMA meta (+78%)"),
        ]
        for i, (name, tflops, note) in enumerate(comp_data):
            row = VGroup(
                Text(name, font_size=12, color=C["fg3"]),
                Text(f"{tflops} TFLOPS", font_size=12, color=C["green_ok"], weight=BOLD),
                Text(note, font_size=11, color=C["blue_tile"]),
            )
            row.arrange(RIGHT, buff=0.8)
            comparison.add(row)
        comparison.arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        comparison.move_to(DOWN * 2.8)
        self.add(comparison)


class ThreeStagePipelineJump(Scene):
    """Fig 3: The 3-stage discontinuity on E4M3 — iter040 crossing 1000 TFLOPS.
    Uses manual bars to avoid LaTeX dependency."""

    def construct(self):
        self.camera.background_color = C["bg"]
        title = Text("E4M3: The 3-Stage Discontinuity", font_size=26, weight=BOLD)
        title.to_edge(UP, buff=0.4)
        self.add(title)

        data = [
            ("Base\n671", 671, C["gray_bg"]),
            ("iter001\n759", 759, C["blue_tile"]),
            ("iter016\n772", 772, C["blue_tile"]),
            ("iter023\n811", 811, C["blue_tile"]),
            ("iter036\n897", 897, C["purple_role"]),
            ("iter040\n1090", 1090, C["green_ok"]),
            ("iter068\n1127", 1127, C["green_ok"]),
        ]

        chart_left = -5.0
        chart_bottom = -2.8
        chart_height = 4.5
        bar_w = 1.0
        min_val = 600
        max_val = 1200

        y_axis = Line(
            [chart_left - 0.1, chart_bottom, 0],
            [chart_left - 0.1, chart_bottom + chart_height + 0.3, 0],
            color=C["fg3"], stroke_width=1.5
        )
        x_axis = Line(
            [chart_left - 0.1, chart_bottom, 0],
            [chart_left + len(data) * (bar_w + 0.2) + 0.3, chart_bottom, 0],
            color=C["fg3"], stroke_width=1.5
        )
        y_label = Text("TFLOPS", font_size=13, color=C["fg3"])
        y_label.next_to(y_axis, LEFT, buff=0.15).rotate(PI / 2)
        self.add(y_axis, x_axis, y_label)

        bars = VGroup()
        for i, (lbl, val, clr) in enumerate(data):
            x = chart_left + i * (bar_w + 0.2) + bar_w / 2
            h = ((val - min_val) / (max_val - min_val)) * chart_height
            bar = Rectangle(
                width=bar_w, height=max(h, 0.05),
                fill_color=clr, fill_opacity=0.8,
                stroke_color=C["fg"], stroke_width=1.5
            )
            bar.move_to([x, chart_bottom + h / 2, 0])

            val_t = Text(str(val), font_size=12, color=clr, weight=BOLD)
            val_t.next_to(bar, UP, buff=0.05)

            lbl_t = Text(lbl, font_size=9, color=C["fg3"])
            lbl_t.next_to(bar, DOWN, buff=0.08)

            bars.add(VGroup(bar, val_t, lbl_t))
        self.add(bars)

        thousand_y = chart_bottom + ((1000 - min_val) / (max_val - min_val)) * chart_height
        thousand_line = DashedLine(
            [chart_left - 0.1, thousand_y, 0],
            [chart_left + len(data) * (bar_w + 0.2) + 0.3, thousand_y, 0],
            color=C["red_accent"], stroke_width=1.5, dash_length=0.15
        )
        thousand_label = Text("1000 TFLOPS", font_size=12, color=C["red_accent"], weight=BOLD)
        thousand_label.next_to(thousand_line, RIGHT, buff=0.1)
        self.add(thousand_line, thousand_label)

        brace_3s = Brace(
            VGroup(bars[4][0], bars[5][0]),
            UP, buff=0.35, color=C["green_ok"]
        )
        brace_label = Text("3-stage\npipeline!", font_size=14, color=C["green_ok"], weight=BOLD)
        brace_label.next_to(brace_3s, UP, buff=0.08)
        self.add(brace_3s, brace_label)
