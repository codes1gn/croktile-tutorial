"""
Part II — Block-Scaled GEMM FP8 figures.
Generates SVG diagrams for the blockscale optimization story.
Run: manim -qh --format=svg blockscale_gemm_figures.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from theme import parse_theme
from manim import *

C, THEME = parse_theme()


class BlockScaleConcept(Scene):
    """Fig 1: Block scaling concept — FP8 blocks with FP32 scale factors."""

    def construct(self):
        self.camera.background_color = C["bg"]
        title = Text("Block Scaling: FP8 Operands + FP32 Scale Factors", font_size=26, weight=BOLD)
        title.to_edge(UP, buff=0.4)
        self.add(title)

        k_label = Text("K dimension →", font_size=14, color=C["fg3"])
        k_label.move_to(LEFT * 1 + UP * 1.6)
        self.add(k_label)

        blocks = VGroup()
        block_labels = ["K-block 0\n(128 elems)", "K-block 1\n(128 elems)", "K-block 2\n(128 elems)"]
        for i, bl in enumerate(block_labels):
            r = RoundedRectangle(
                width=2.5, height=1.2, corner_radius=0.1,
                fill_color=C["blue_tile"], fill_opacity=0.6,
                stroke_width=1.5
            )
            t = Text(bl, font_size=12, color=C["fg"])
            dtype_badge = Text("FP8 E4M3", font_size=9, color=C["gold"], weight=BOLD)
            t.move_to(r.get_center() + UP * 0.1)
            dtype_badge.move_to(r.get_center() + DOWN * 0.35)
            blocks.add(VGroup(r, t, dtype_badge))
        blocks.arrange(RIGHT, buff=0.15)
        blocks.move_to(UP * 0.5)
        self.add(blocks)

        scales = VGroup()
        for i in range(3):
            sr = RoundedRectangle(
                width=1.0, height=0.5, corner_radius=0.08,
                fill_color=C["orange_tile"], fill_opacity=0.7,
                stroke_color=C["orange_tile"], stroke_width=1.5
            )
            st = Text(f"s{i}", font_size=12, color=C["fg"], weight=BOLD)
            st_dtype = Text("FP32", font_size=8, color=C["fg"])
            st.move_to(sr.get_center() + UP * 0.05)
            st_dtype.move_to(sr.get_center() + DOWN * 0.15)
            scale_g = VGroup(sr, st, st_dtype)
            scale_g.next_to(blocks[i], DOWN, buff=0.15)
            scales.add(scale_g)
        self.add(scales)

        for i in range(3):
            arr = Arrow(
                scales[i].get_bottom(), scales[i].get_bottom() + DOWN * 0.6,
                buff=0, color=C["orange_tile"], stroke_width=1.5, max_tip_length_to_length_ratio=0.3
            )
            self.add(arr)

        formula_box = RoundedRectangle(
            width=8, height=0.9, corner_radius=0.1,
            fill_color=C["purple_role"], fill_opacity=0.15,
            stroke_color=C["purple_role"], stroke_width=1.5
        )
        formula = Text(
            "Inner product block b:  s_a × s_b × ⟨ã, b̃⟩  →  accumulate in FP16",
            font_size=14, color=C["purple_role"]
        )
        formula.move_to(formula_box)
        fg = VGroup(formula_box, formula)
        fg.move_to(DOWN * 1.8)
        self.add(fg)

        tradeoff = Text(
            "Every K-tile pulls matrix data AND scale metadata → extra loads on critical path",
            font_size=13, color=C["red_accent"], slant=ITALIC
        )
        tradeoff.to_edge(DOWN, buff=0.3)
        self.add(tradeoff)


class TMAOverlap(Scene):
    """Fig 2: iter049 — TMA overlap with scale accumulation."""

    def construct(self):
        self.camera.background_color = C["bg"]
        title = Text("Step 1: TMA Overlap with Scale Accumulation (iter049)", font_size=24, weight=BOLD)
        title.to_edge(UP, buff=0.4)
        self.add(title)

        result = Text("380 TFLOPS @2048³ (+21%)", font_size=18, color=C["green_ok"])
        result.next_to(title, DOWN, buff=0.15)
        self.add(result)

        time_label = Text("Time →", font_size=14, color=C["fg3"])
        time_label.move_to(LEFT * 5.5 + UP * 0.8)
        self.add(time_label)

        def make_timeline(name, color, slots, y):
            label = Text(name, font_size=14, color=color, weight=BOLD)
            label.move_to(LEFT * 5.5 + UP * y)
            blocks = VGroup()
            for txt, clr, op, w in slots:
                r = Rectangle(width=w, height=0.45, fill_color=clr, fill_opacity=op, stroke_width=1)
                t = Text(txt, font_size=8, color=C["fg"])
                t.move_to(r)
                blocks.add(VGroup(r, t))
            blocks.arrange(RIGHT, buff=0.04)
            blocks.next_to(label, RIGHT, buff=0.3)
            return VGroup(label, blocks)

        before_label = Text("Before (baseline):", font_size=14, color=C["red_accent"])
        before_label.move_to(LEFT * 4 + UP * 0.3)
        self.add(before_label)

        before_slots = [
            ("WGMMA", C["purple_role"], 0.8, 1.2),
            ("scale\naccum", C["orange_tile"], 0.8, 0.9),
            ("TMA\nnext K", C["blue_tile"], 0.8, 1.0),
            ("wait...", C["gray_bg"], 0.3, 0.8),
            ("WGMMA", C["purple_role"], 0.8, 1.2),
        ]
        before_tl = make_timeline("K iter", C["red_accent"], before_slots, -0.3)
        self.add(before_tl)

        idle_brace = Brace(before_tl[1][3][0], DOWN, buff=0.05, color=C["red_accent"])
        idle_text = Text("TMA idle!", font_size=10, color=C["red_accent"])
        idle_text.next_to(idle_brace, DOWN, buff=0.05)
        self.add(idle_brace, idle_text)

        after_label = Text("After (iter049):", font_size=14, color=C["green_ok"])
        after_label.move_to(LEFT * 4 + DOWN * 1.5)
        self.add(after_label)

        after_slots = [
            ("WGMMA", C["purple_role"], 0.8, 1.2),
            ("TMA next K\n+ scale accum", C["teal"], 0.8, 1.5),
            ("WGMMA", C["purple_role"], 0.8, 1.2),
            ("TMA next K\n+ scale accum", C["teal"], 0.8, 1.5),
        ]
        after_tl = make_timeline("K iter", C["green_ok"], after_slots, -2.1)
        self.add(after_tl)

        overlap_note = Text(
            "Overlap independent work: TMA starts while scale_accumulator runs",
            font_size=13, color=C["teal"], slant=ITALIC
        )
        overlap_note.to_edge(DOWN, buff=0.3)
        self.add(overlap_note)


class N256VsN128(Scene):
    """Fig 3: N256 tile — more math per CTA, fewer CTAs."""

    def construct(self):
        self.camera.background_color = C["bg"]
        title = Text("Step 2: N256 WGMMA — Double Math Per Tile", font_size=24, weight=BOLD)
        title.to_edge(UP, buff=0.4)
        self.add(title)

        n128_label = Text("N128 (baseline)", font_size=16, color=C["red_accent"], weight=BOLD)
        n128_label.move_to(LEFT * 3.5 + UP * 1.3)
        self.add(n128_label)

        n128_tiles = VGroup()
        for i in range(4):
            for j in range(2):
                r = Rectangle(
                    width=0.8, height=0.6,
                    fill_color=C["blue_tile"], fill_opacity=0.5,
                    stroke_width=1, stroke_color=C["fg"]
                )
                r.move_to(LEFT * 4.5 + RIGHT * i * 0.85 + DOWN * j * 0.65 + UP * 0.3)
                n128_tiles.add(r)
        self.add(n128_tiles)

        n128_note = Text("8 CTAs, each M64×N128", font_size=12, color=C["fg3"])
        n128_note.next_to(n128_tiles, DOWN, buff=0.2)
        self.add(n128_note)

        n256_label = Text("N256 (iter051)", font_size=16, color=C["green_ok"], weight=BOLD)
        n256_label.move_to(RIGHT * 3.5 + UP * 1.3)
        self.add(n256_label)

        n256_tiles = VGroup()
        for i in range(2):
            for j in range(2):
                r = Rectangle(
                    width=1.6, height=0.6,
                    fill_color=C["green_ok"], fill_opacity=0.6,
                    stroke_width=1, stroke_color=C["fg"]
                )
                r.move_to(RIGHT * 3 + RIGHT * i * 1.7 + DOWN * j * 0.65 + UP * 0.3)
                n256_tiles.add(r)
        self.add(n256_tiles)

        n256_note = Text("4 CTAs, each M64×N256 (2× math)", font_size=12, color=C["fg3"])
        n256_note.next_to(n256_tiles, DOWN, buff=0.2)
        self.add(n256_note)

        tradeoff_box = RoundedRectangle(
            width=10, height=1.2, corner_radius=0.1,
            fill_color=C["orange_tile"], fill_opacity=0.1,
            stroke_color=C["orange_tile"], stroke_width=1
        )
        tradeoff_text = VGroup(
            Text("@4096³: 602 TFLOPS (+51%) — fewer CTAs, more math per CTA ✓", font_size=13, color=C["green_ok"]),
            Text("@2048³: 372 TFLOPS (−2%) — grid too coarse for small problem ✗", font_size=13, color=C["red_accent"]),
        )
        tradeoff_text.arrange(DOWN, buff=0.1)
        tradeoff_text.move_to(tradeoff_box)
        tg = VGroup(tradeoff_box, tradeoff_text)
        tg.move_to(DOWN * 2.0)
        self.add(tg)

        smem_note = Text("SMEM ≈ 40 KB per stage (N256) — workable on Hopper", font_size=13, color=C["fg3"])
        smem_note.to_edge(DOWN, buff=0.3)
        self.add(smem_note)


class OptimizationLadder(Scene):
    """Fig 4: Full optimization ladder as a bar chart."""

    def construct(self):
        self.camera.background_color = C["bg"]
        title = Text("Block-Scaled GEMM: Optimization Ladder @4096³", font_size=24, weight=BOLD)
        title.to_edge(UP, buff=0.4)
        self.add(title)

        data = [
            ("Baseline\nM64N128K32", 397.9, C["gray_bg"]),
            ("iter049\nTMA overlap", 380, C["orange_tile"]),
            ("iter051\nN256", 602, C["blue_tile"]),
            ("iter053\nN256+L2", 610, C["teal"]),
            ("iter066\nN256+L2\n+prefetch", 621, C["green_ok"]),
        ]

        max_val = 650
        bar_width = 1.3
        chart_width = len(data) * (bar_width + 0.3)
        start_x = -chart_width / 2 + bar_width / 2

        bars = VGroup()
        for i, (label, val, color) in enumerate(data):
            x = start_x + i * (bar_width + 0.3)
            height = (val / max_val) * 4.0
            bar = Rectangle(
                width=bar_width, height=height,
                fill_color=color, fill_opacity=0.8,
                stroke_width=1.5, stroke_color=C["fg"]
            )
            bar.move_to(RIGHT * x + DOWN * (2.0 - height / 2))

            val_text = Text(f"{val}", font_size=14, color=color, weight=BOLD)
            val_text.next_to(bar, UP, buff=0.08)

            label_text = Text(label, font_size=10, color=C["fg3"])
            label_text.next_to(bar, DOWN, buff=0.1)

            bars.add(VGroup(bar, val_text, label_text))
        self.add(bars)

        note_text = Text(
            "iter049 measured @2048³ (380), not 4096³ — different size tradeoff",
            font_size=11, color=C["orange_tile"], slant=ITALIC
        )
        note_text.to_edge(DOWN, buff=0.15)
        self.add(note_text)

        delta_text = Text("+56% from baseline to best @4096³", font_size=16, color=C["green_ok"], weight=BOLD)
        delta_text.move_to(UP * 1.0 + RIGHT * 2)
        self.add(delta_text)
