"""
Figure 1: Per-Element vs Data-Block Programming Model
Side-by-side comparison showing the mental model difference.

Left: SIMD per-element view — each thread touches one cell, many scattered arrows
Right: Block-level view — one DMA moves an entire tile, single arrow to block
"""
from manim import *

TILE_COLOR = "#4CAF50"
ELEM_COLOR = "#FF9800"
ARROW_COLOR = "#2196F3"
MEM_GLOBAL = "#37474F"
MEM_LOCAL = "#1B5E20"
BG_COLOR = "#1a1a2e"
GRID_FILL = "#263238"
HIGHLIGHT = "#FFEB3B"


class ElementVsBlock(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        title_left = Text("Per-Element View", font_size=28, color=ELEM_COLOR, font="Monospace")
        title_right = Text("Data-Block View", font_size=28, color=TILE_COLOR, font="Monospace")
        title_left.move_to(LEFT * 3.5 + UP * 3.2)
        title_right.move_to(RIGHT * 3.5 + UP * 3.2)

        subtitle_left = Text("(CUDA / SIMD style)", font_size=18, color=GREY_B, font="Monospace")
        subtitle_right = Text("(Croktile style)", font_size=18, color=GREY_B, font="Monospace")
        subtitle_left.next_to(title_left, DOWN, buff=0.15)
        subtitle_right.next_to(title_right, DOWN, buff=0.15)

        sep = DashedLine(UP * 3.5, DOWN * 3.5, color=GREY_D, dash_length=0.1)

        self.add(title_left, title_right, subtitle_left, subtitle_right, sep)

        # --- Left side: per-element ---
        left_origin = LEFT * 3.5 + UP * 1.2

        global_label_l = Text("Global Memory", font_size=16, color=GREY_B, font="Monospace")
        global_label_l.move_to(left_origin + UP * 0.5)

        cells_l = VGroup()
        for i in range(8):
            r = Rectangle(width=0.55, height=0.55, fill_color=GRID_FILL,
                          fill_opacity=1, stroke_color=GREY_C, stroke_width=1)
            r.move_to(left_origin + RIGHT * (i - 3.5) * 0.6)
            idx = Text(str(i), font_size=12, color=GREY_B, font="Monospace").move_to(r)
            cells_l.add(VGroup(r, idx))

        thread_label_l = Text("Threads", font_size=16, color=GREY_B, font="Monospace")
        thread_label_l.move_to(left_origin + DOWN * 2.2)

        threads_l = VGroup()
        arrows_l = VGroup()
        for i in range(8):
            t = Circle(radius=0.18, fill_color=ELEM_COLOR, fill_opacity=0.8,
                       stroke_color=WHITE, stroke_width=1)
            t.move_to(left_origin + DOWN * 2.8 + RIGHT * (i - 3.5) * 0.6)
            tid = Text(f"t{i}", font_size=10, color=WHITE, font="Monospace").move_to(t)
            threads_l.add(VGroup(t, tid))

            arr = Arrow(
                start=cells_l[i][0].get_bottom(),
                end=t.get_top(),
                buff=0.08, stroke_width=1.5, color=ELEM_COLOR,
                max_tip_length_to_length_ratio=0.15
            )
            arrows_l.add(arr)

        code_l = Text(
            "parallel {i} by [8]\n  out.at(i) = a.at(i) + b.at(i);",
            font_size=13, color=GREY_A, font="Monospace"
        ).move_to(left_origin + DOWN * 4.2)

        desc_l = Text(
            "8 threads, 8 individual reads\neach thread: 1 element",
            font_size=12, color=GREY_C, font="Monospace"
        ).move_to(left_origin + DOWN * 5.5)

        self.add(global_label_l, cells_l, thread_label_l, threads_l, arrows_l, code_l, desc_l)

        # --- Right side: data-block ---
        right_origin = RIGHT * 3.5 + UP * 1.2

        global_label_r = Text("Global Memory", font_size=16, color=GREY_B, font="Monospace")
        global_label_r.move_to(right_origin + UP * 0.5)

        tile_r = VGroup()
        for i in range(8):
            r = Rectangle(width=0.55, height=0.55, fill_color=TILE_COLOR,
                          fill_opacity=0.35, stroke_color=TILE_COLOR, stroke_width=1.5)
            r.move_to(right_origin + RIGHT * (i - 3.5) * 0.6)
            idx = Text(str(i), font_size=12, color=WHITE, font="Monospace").move_to(r)
            tile_r.add(VGroup(r, idx))

        tile_bracket = Brace(tile_r, DOWN, buff=0.1, color=TILE_COLOR)
        tile_label = Text("1 tile (8 elements)", font_size=13, color=TILE_COLOR, font="Monospace")
        tile_label.next_to(tile_bracket, DOWN, buff=0.1)

        local_label = Text("Local Memory", font_size=16, color=MEM_LOCAL, font="Monospace")
        local_label.move_to(right_origin + DOWN * 2.0)

        local_tile = VGroup()
        for i in range(8):
            r = Rectangle(width=0.55, height=0.55, fill_color=MEM_LOCAL,
                          fill_opacity=0.4, stroke_color=TILE_COLOR, stroke_width=1.5)
            r.move_to(right_origin + DOWN * 2.7 + RIGHT * (i - 3.5) * 0.6)
            idx = Text(str(i), font_size=12, color=WHITE, font="Monospace").move_to(r)
            local_tile.add(VGroup(r, idx))

        dma_arrow = Arrow(
            start=right_origin + DOWN * 0.3,
            end=right_origin + DOWN * 2.1,
            buff=0.15, stroke_width=3, color=ARROW_COLOR,
            max_tip_length_to_length_ratio=0.1
        )
        dma_label = Text("dma.copy", font_size=14, color=ARROW_COLOR, font="Monospace")
        dma_label.next_to(dma_arrow, RIGHT, buff=0.15)

        code_r = Text(
            "parallel tile by N {\n  f = dma.copy a.chunkat(tile) => local;\n  // compute on f.data\n}",
            font_size=13, color=GREY_A, font="Monospace"
        ).move_to(right_origin + DOWN * 4.4)

        desc_r = Text(
            "1 DMA, 1 bulk transfer\nentire tile moves at once",
            font_size=12, color=GREY_C, font="Monospace"
        ).move_to(right_origin + DOWN * 5.7)

        self.add(global_label_r, tile_r, tile_bracket, tile_label,
                 local_label, local_tile, dma_arrow, dma_label, code_r, desc_r)


if __name__ == "__main__":
    import subprocess, sys
    subprocess.run([
        sys.executable, "-m", "manim", "render",
        "-ql", "--format=png", "-o", "ch02_fig1_element_vs_block",
        __file__, "ElementVsBlock"
    ])
