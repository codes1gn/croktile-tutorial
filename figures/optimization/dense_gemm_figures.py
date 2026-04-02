"""
Part II — Dense GEMM FP16 figures.
Generates SVG diagrams for each kernel variant in the optimization story.
Run: manim -qh --format=svg dense_gemm_figures.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from theme import parse_theme
from manim import *

C, THEME = parse_theme()


class BaselineKernel(Scene):
    """Fig 1: Baseline 1p1c pipeline — producer/consumer with 4-stage ring,
    showing the bubble where consumer waits."""

    def construct(self):
        self.camera.background_color = C["bg"]
        title = Text("Baseline: 1p1c, WN=128, 4-Stage Ring", font_size=28, weight=BOLD)
        title.to_edge(UP, buff=0.4)
        self.add(title)

        subtitle = Text("208.7 TFLOPS — consumer stalls on wait_full", font_size=18, color=C["fg3"])
        subtitle.next_to(title, DOWN, buff=0.15)
        self.add(subtitle)

        stages_label = Text("Shared Memory Ring (4 stages)", font_size=16, color=C["fg3"])
        stages_label.move_to(UP * 1.2)
        self.add(stages_label)

        stage_colors = [C["blue_tile"], C["blue_tile"], C["gray_bg"], C["gray_bg"]]
        stage_labels_text = ["S0\nfull", "S1\nfull", "S2\nempty", "S3\nempty"]
        stages = VGroup()
        for i in range(4):
            r = RoundedRectangle(
                width=1.3, height=0.9, corner_radius=0.1,
                fill_color=stage_colors[i], fill_opacity=0.7,
                stroke_color=C["fg"], stroke_width=1.5
            )
            t = Text(stage_labels_text[i], font_size=12, color=C["fg"])
            t.move_to(r.get_center())
            stages.add(VGroup(r, t))
        stages.arrange(RIGHT, buff=0.3)
        stages.move_to(UP * 0.3)
        self.add(stages)

        producer_box = RoundedRectangle(
            width=2.5, height=1.0, corner_radius=0.15,
            fill_color=C["orange_tile"], fill_opacity=0.8,
            stroke_width=2
        )
        producer_label = Text("TMA Producer\n(1 warpgroup)", font_size=14, color=C["fg"])
        producer_label.move_to(producer_box)
        producer = VGroup(producer_box, producer_label)
        producer.move_to(DOWN * 1.5 + LEFT * 2.5)
        self.add(producer)

        consumer_box = RoundedRectangle(
            width=2.5, height=1.0, corner_radius=0.15,
            fill_color=C["purple_role"], fill_opacity=0.8,
            stroke_width=2
        )
        consumer_label = Text("WGMMA Consumer\n(1 warpgroup)", font_size=14, color=C["fg"])
        consumer_label.move_to(consumer_box)
        consumer = VGroup(consumer_box, consumer_label)
        consumer.move_to(DOWN * 1.5 + RIGHT * 2.5)
        self.add(consumer)

        arr1 = Arrow(
            producer.get_top(), stages[0].get_bottom(),
            buff=0.15, color=C["orange_tile"], stroke_width=2.5
        )
        arr1_label = Text("TMA load", font_size=11, color=C["orange_tile"])
        arr1_label.next_to(arr1, LEFT, buff=0.1)
        self.add(arr1, arr1_label)

        arr2 = Arrow(
            stages[1].get_bottom(), consumer.get_top(),
            buff=0.15, color=C["purple_role"], stroke_width=2.5
        )
        arr2_label = Text("MMA consume", font_size=11, color=C["purple_role"])
        arr2_label.next_to(arr2, RIGHT, buff=0.1)
        self.add(arr2, arr2_label)

        bubble = RoundedRectangle(
            width=4.5, height=0.5, corner_radius=0.1,
            fill_color=C["red_accent"], fill_opacity=0.3,
            stroke_color=C["red_accent"], stroke_width=1.5
        )
        bubble_text = Text("⚠ Pipeline bubble: producer stalls on wait_empty", font_size=12, color=C["red_accent"])
        bubble_group = VGroup(bubble, bubble_text)
        bubble_text.move_to(bubble)
        bubble_group.move_to(DOWN * 2.8)
        self.add(bubble_group)

        smem_note = Text(
            "SMEM ≈ 4 × (64×64 + 128×64) × 2B ≈ 96 KB → 2 CTAs/SM",
            font_size=13, color=C["fg3"]
        )
        smem_note.to_edge(DOWN, buff=0.3)
        self.add(smem_note)


class Step2ThreeStage(Scene):
    """Fig 2: 3-stage pipeline with WN=176 — producer runs ahead."""

    def construct(self):
        self.camera.background_color = C["bg"]
        title = Text("Step 2: 3-Stage Pipeline (WN=176)", font_size=28, weight=BOLD)
        title.to_edge(UP, buff=0.4)
        self.add(title)

        result = Text("354.1 TFLOPS (+46%) — producer runs ahead of consumer", font_size=18, color=C["green_ok"])
        result.next_to(title, DOWN, buff=0.15)
        self.add(result)

        time_label = Text("Time →", font_size=14, color=C["fg3"])
        time_label.move_to(LEFT * 5.5 + UP * 0.5)
        self.add(time_label)

        def make_timeline(name, color, slots, y_pos):
            label = Text(name, font_size=14, color=color, weight=BOLD)
            label.move_to(LEFT * 5.2 + UP * y_pos)
            blocks = VGroup()
            for i, (text, fill_color, opacity) in enumerate(slots):
                r = Rectangle(
                    width=1.2, height=0.55,
                    fill_color=fill_color, fill_opacity=opacity,
                    stroke_color=C["fg"], stroke_width=1
                )
                t = Text(text, font_size=10, color=C["fg"])
                t.move_to(r)
                blocks.add(VGroup(r, t))
            blocks.arrange(RIGHT, buff=0.08)
            blocks.move_to(RIGHT * 0.5 + UP * y_pos)
            return VGroup(label, blocks)

        producer_slots = [
            ("TMA S0", C["orange_tile"], 0.8),
            ("TMA S1", C["orange_tile"], 0.8),
            ("TMA S2", C["orange_tile"], 0.8),
            ("TMA S0", C["orange_tile"], 0.6),
            ("TMA S1", C["orange_tile"], 0.6),
            ("TMA S2", C["orange_tile"], 0.6),
            ("TMA S0", C["orange_tile"], 0.4),
        ]
        consumer_slots = [
            ("wait", C["gray_bg"], 0.3),
            ("wait", C["gray_bg"], 0.3),
            ("MMA S0", C["purple_role"], 0.8),
            ("MMA S1", C["purple_role"], 0.8),
            ("MMA S2", C["purple_role"], 0.8),
            ("MMA S0", C["purple_role"], 0.6),
            ("MMA S1", C["purple_role"], 0.6),
        ]

        prod_tl = make_timeline("Producer", C["orange_tile"], producer_slots, 0)
        cons_tl = make_timeline("Consumer", C["purple_role"], consumer_slots, -1.0)
        self.add(prod_tl, cons_tl)

        brace = Brace(
            VGroup(prod_tl[1][0], prod_tl[1][1]),
            UP, buff=0.1, color=C["green_ok"]
        )
        brace_text = Text("Producer 1 stage ahead", font_size=11, color=C["green_ok"])
        brace_text.next_to(brace, UP, buff=0.08)
        self.add(brace, brace_text)

        smem_box = RoundedRectangle(
            width=8, height=0.5, corner_radius=0.1,
            fill_color=C["blue_tile"], fill_opacity=0.15,
            stroke_color=C["blue_tile"], stroke_width=1
        )
        smem_text = Text(
            "SMEM ≈ 3 × (64×64 + 176×64) × 2B ≈ 90 KB → still 2 CTAs/SM ✓",
            font_size=13, color=C["blue_tile"]
        )
        smem_text.move_to(smem_box)
        smem_group = VGroup(smem_box, smem_text)
        smem_group.move_to(DOWN * 2.3)
        self.add(smem_group)

        key_insight = Text(
            "Key: extra stage bought producer-consumer concurrency, not more math",
            font_size=14, color=C["fg3"], slant=ITALIC
        )
        key_insight.to_edge(DOWN, buff=0.3)
        self.add(key_insight)


class SplitOutput1p2c(Scene):
    """Fig 3: Split-output 1p2c — two consumers with private output slices."""

    def construct(self):
        self.camera.background_color = C["bg"]
        title = Text("Step 3–4: Split-Output 1p2c", font_size=28, weight=BOLD)
        title.to_edge(UP, buff=0.4)
        self.add(title)

        result = Text("382.5 TFLOPS — output contention eliminated", font_size=18, color=C["green_ok"])
        result.next_to(title, DOWN, buff=0.15)
        self.add(result)

        producer = RoundedRectangle(
            width=2.2, height=1.4, corner_radius=0.15,
            fill_color=C["orange_tile"], fill_opacity=0.8
        )
        p_label = Text("TMA\nProducer", font_size=14, color=C["fg"])
        p_label.move_to(producer)
        prod_g = VGroup(producer, p_label)
        prod_g.move_to(LEFT * 4 + DOWN * 0.3)
        self.add(prod_g)

        smem = RoundedRectangle(
            width=2.5, height=2.5, corner_radius=0.15,
            fill_color=C["blue_tile"], fill_opacity=0.2,
            stroke_color=C["blue_tile"], stroke_width=2
        )
        smem_label = Text("SMEM\nOperand\nRing", font_size=13, color=C["blue_tile"])
        smem_label.move_to(smem)
        smem_g = VGroup(smem, smem_label)
        smem_g.move_to(LEFT * 0.8 + DOWN * 0.3)
        self.add(smem_g)

        c1_box = RoundedRectangle(
            width=2.0, height=1.0, corner_radius=0.1,
            fill_color=C["purple_role"], fill_opacity=0.8
        )
        c1_label = Text("Consumer 1\n(WGMMA)", font_size=12, color=C["fg"])
        c1_label.move_to(c1_box)
        c1 = VGroup(c1_box, c1_label)
        c1.move_to(RIGHT * 2.5 + UP * 0.5)

        c2_box = RoundedRectangle(
            width=2.0, height=1.0, corner_radius=0.1,
            fill_color=C["purple_role"], fill_opacity=0.8
        )
        c2_label = Text("Consumer 2\n(WGMMA)", font_size=12, color=C["fg"])
        c2_label.move_to(c2_box)
        c2 = VGroup(c2_box, c2_label)
        c2.move_to(RIGHT * 2.5 + DOWN * 1.1)
        self.add(c1, c2)

        out1 = RoundedRectangle(
            width=1.5, height=0.7, corner_radius=0.1,
            fill_color=C["green_ok"], fill_opacity=0.6
        )
        out1_t = Text("Output\nSlice A", font_size=10, color=C["fg"])
        out1_t.move_to(out1)
        out1_g = VGroup(out1, out1_t)
        out1_g.move_to(RIGHT * 5.0 + UP * 0.5)

        out2 = RoundedRectangle(
            width=1.5, height=0.7, corner_radius=0.1,
            fill_color=C["green_ok"], fill_opacity=0.6
        )
        out2_t = Text("Output\nSlice B", font_size=10, color=C["fg"])
        out2_t.move_to(out2)
        out2_g = VGroup(out2, out2_t)
        out2_g.move_to(RIGHT * 5.0 + DOWN * 1.1)
        self.add(out1_g, out2_g)

        self.add(Arrow(prod_g.get_right(), smem_g.get_left(), buff=0.15, color=C["orange_tile"], stroke_width=2))
        self.add(Arrow(smem_g.get_right(), c1.get_left(), buff=0.15, color=C["blue_tile"], stroke_width=2))
        self.add(Arrow(smem_g.get_right(), c2.get_left(), buff=0.15, color=C["blue_tile"], stroke_width=2))
        self.add(Arrow(c1.get_right(), out1_g.get_left(), buff=0.1, color=C["green_ok"], stroke_width=2))
        self.add(Arrow(c2.get_right(), out2_g.get_left(), buff=0.1, color=C["green_ok"], stroke_width=2))

        vs_box = RoundedRectangle(
            width=9.5, height=0.8, corner_radius=0.1,
            fill_color=C["red_accent"], fill_opacity=0.1,
            stroke_color=C["red_accent"], stroke_width=1
        )
        vs_text = Text(
            "1p1c: both consumers share ONE output_s tile → serialization\n"
            "1p2c split-output: each consumer gets PRIVATE output slice → no contention",
            font_size=12, color=C["red_accent"], line_spacing=1.2
        )
        vs_text.move_to(vs_box)
        vs_g = VGroup(vs_box, vs_text)
        vs_g.move_to(DOWN * 2.8)
        self.add(vs_g)


class OccupancyCliff(Scene):
    """Fig 4: WN sweep showing the occupancy cliff at WN=168.
    Uses manual bars instead of Axes to avoid LaTeX dependency."""

    def construct(self):
        self.camera.background_color = C["bg"]
        title = Text("The WN=168 Occupancy Cliff", font_size=28, weight=BOLD)
        title.to_edge(UP, buff=0.4)
        self.add(title)

        data = [
            ("128", 350, C["green_ok"]), ("136", 360, C["green_ok"]),
            ("144", 370, C["green_ok"]), ("152", 382, C["green_ok"]),
            ("160", 380, C["green_ok"]), ("168", 150, C["red_accent"]),
            ("176", 140, C["red_accent"]),
        ]

        max_val = 420
        bar_w = 0.85
        chart_left = -4.0
        chart_bottom = -2.5
        chart_height = 4.0

        y_axis = Line(
            start=[chart_left - 0.1, chart_bottom, 0],
            end=[chart_left - 0.1, chart_bottom + chart_height + 0.3, 0],
            color=C["fg3"], stroke_width=1.5
        )
        x_axis = Line(
            start=[chart_left - 0.1, chart_bottom, 0],
            end=[chart_left + len(data) * (bar_w + 0.2) + 0.3, chart_bottom, 0],
            color=C["fg3"], stroke_width=1.5
        )
        self.add(y_axis, x_axis)

        y_label = Text("TFLOPS @8192³", font_size=13, color=C["fg3"])
        y_label.next_to(y_axis, LEFT, buff=0.15).rotate(PI / 2)
        x_label = Text("WN (MATMUL_WARP_N)", font_size=13, color=C["fg3"])
        x_label.next_to(x_axis, DOWN, buff=0.25)
        self.add(y_label, x_label)

        bars = VGroup()
        for i, (wn, val, clr) in enumerate(data):
            x = chart_left + i * (bar_w + 0.2) + bar_w / 2
            h = (val / max_val) * chart_height
            bar = Rectangle(
                width=bar_w, height=h,
                fill_color=clr, fill_opacity=0.8,
                stroke_color=C["fg"], stroke_width=1.5
            )
            bar.move_to([x, chart_bottom + h / 2, 0])

            val_t = Text(str(val), font_size=12, color=clr, weight=BOLD)
            val_t.next_to(bar, UP, buff=0.06)

            wn_t = Text(wn, font_size=11, color=C["fg3"])
            wn_t.next_to(bar, DOWN, buff=0.06)

            bars.add(VGroup(bar, val_t, wn_t))
        self.add(bars)

        cliff_x = chart_left + 4.5 * (bar_w + 0.2)
        cliff_line = DashedLine(
            [cliff_x, chart_bottom, 0], [cliff_x, chart_bottom + chart_height + 0.2, 0],
            color=C["red_accent"], stroke_width=2, dash_length=0.12
        )
        cliff_label = Text("228 KB\nSMEM limit", font_size=12, color=C["red_accent"], weight=BOLD)
        cliff_label.next_to(cliff_line, RIGHT, buff=0.15).shift(UP * 0.3)
        self.add(cliff_line, cliff_label)

        ann_2cta = Text("2 CTAs/SM", font_size=14, color=C["green_ok"], weight=BOLD)
        ann_2cta.move_to([chart_left + 1.5, chart_bottom + chart_height - 0.3, 0])
        ann_1cta = Text("1 CTA/SM", font_size=14, color=C["red_accent"], weight=BOLD)
        ann_1cta.move_to([chart_left + 5.5, chart_bottom + chart_height - 1.5, 0])
        self.add(ann_2cta, ann_1cta)
