"""
Animation 1: Per-Element vs Data-Block — animated version.
Shows scattered per-element arrows firing one by one,
then contrasts with a single bulk DMA transfer.
"""
from manim import *

BG = "#1a1a2e"
TILE_C = "#4CAF50"
ELEM_C = "#FF9800"
ARROW_C = "#2196F3"
LOCAL_C = "#1B5E20"


class ElementVsBlockAnim(Scene):
    def construct(self):
        self.camera.background_color = BG

        # --- Phase 1: Per-Element ---
        phase1_title = Text("Per-Element View", font_size=32, color=ELEM_C, font="Monospace")
        phase1_title.to_edge(UP, buff=0.4)
        self.play(Write(phase1_title), run_time=0.6)

        mem_label = Text("Global Memory", font_size=18, color=GREY_B, font="Monospace")
        mem_label.move_to(UP * 2)

        cells = VGroup()
        for i in range(8):
            r = Rectangle(width=0.7, height=0.55, fill_color="#263238",
                          fill_opacity=1, stroke_color=GREY_C, stroke_width=1)
            r.move_to(LEFT * 2.45 + RIGHT * i * 0.75 + UP * 1.3)
            idx = Text(str(i), font_size=14, color=GREY_B, font="Monospace").move_to(r)
            cells.add(VGroup(r, idx))

        self.play(FadeIn(mem_label), *[FadeIn(c) for c in cells], run_time=0.5)

        thread_label = Text("Threads", font_size=18, color=GREY_B, font="Monospace")
        thread_label.move_to(DOWN * 0.5)

        threads = VGroup()
        for i in range(8):
            t = Circle(radius=0.22, fill_color=ELEM_C, fill_opacity=0.8,
                       stroke_color=WHITE, stroke_width=1)
            t.move_to(LEFT * 2.45 + RIGHT * i * 0.75 + DOWN * 1.2)
            tid = Text(f"t{i}", font_size=11, color=WHITE, font="Monospace").move_to(t)
            threads.add(VGroup(t, tid))

        self.play(FadeIn(thread_label), *[FadeIn(t) for t in threads], run_time=0.5)

        arrows = VGroup()
        for i in range(8):
            arr = Arrow(cells[i][0].get_bottom(), threads[i][0].get_top(),
                        buff=0.08, stroke_width=2, color=ELEM_C,
                        max_tip_length_to_length_ratio=0.15)
            arrows.add(arr)

        for i in range(8):
            cells[i][0].save_state()
            self.play(
                cells[i][0].animate.set_fill(ELEM_C, 0.6),
                GrowArrow(arrows[i]),
                run_time=0.15
            )

        note1 = Text("8 threads = 8 individual transfers", font_size=16,
                      color=GREY_C, font="Monospace")
        note1.move_to(DOWN * 2.5)
        self.play(FadeIn(note1), run_time=0.3)
        self.wait(1.5)

        # Clear phase 1
        self.play(
            *[FadeOut(m) for m in [phase1_title, mem_label, cells, thread_label,
                                    threads, arrows, note1]],
            run_time=0.5
        )

        # --- Phase 2: Data-Block ---
        phase2_title = Text("Data-Block View", font_size=32, color=TILE_C, font="Monospace")
        phase2_title.to_edge(UP, buff=0.4)
        self.play(Write(phase2_title), run_time=0.6)

        gmem_label = Text("Global Memory", font_size=18, color=GREY_B, font="Monospace")
        gmem_label.move_to(UP * 2)

        tile_cells = VGroup()
        for i in range(8):
            r = Rectangle(width=0.7, height=0.55, fill_color=TILE_C,
                          fill_opacity=0.3, stroke_color=TILE_C, stroke_width=1.5)
            r.move_to(LEFT * 2.45 + RIGHT * i * 0.75 + UP * 1.3)
            idx = Text(str(i), font_size=14, color=WHITE, font="Monospace").move_to(r)
            tile_cells.add(VGroup(r, idx))

        self.play(FadeIn(gmem_label), *[FadeIn(c) for c in tile_cells], run_time=0.5)

        lmem_label = Text("Local Memory", font_size=18, color=LOCAL_C, font="Monospace")
        lmem_label.move_to(DOWN * 0.5)

        local_cells = VGroup()
        for i in range(8):
            r = Rectangle(width=0.7, height=0.55, fill_color=LOCAL_C,
                          fill_opacity=0.3, stroke_color=TILE_C, stroke_width=1.5)
            r.move_to(LEFT * 2.45 + RIGHT * i * 0.75 + DOWN * 1.2)
            idx = Text(str(i), font_size=14, color=WHITE, font="Monospace").move_to(r)
            local_cells.add(VGroup(r, idx))

        self.play(FadeIn(lmem_label), *[FadeIn(c) for c in local_cells], run_time=0.5)

        # Single big arrow
        big_arrow = Arrow(UP * 0.8, DOWN * 0.2,
                          buff=0, stroke_width=5, color=ARROW_C,
                          max_tip_length_to_length_ratio=0.08)
        dma_label = Text("dma.copy", font_size=18, color=ARROW_C, font="Monospace")
        dma_label.next_to(big_arrow, RIGHT, buff=0.3)

        # Highlight all cells simultaneously
        self.play(
            *[tile_cells[i][0].animate.set_fill(TILE_C, 0.7) for i in range(8)],
            GrowArrow(big_arrow),
            FadeIn(dma_label),
            run_time=0.8
        )
        self.play(
            *[local_cells[i][0].animate.set_fill(TILE_C, 0.5) for i in range(8)],
            run_time=0.5
        )

        note2 = Text("1 DMA = entire tile moves at once", font_size=16,
                      color=GREY_C, font="Monospace")
        note2.move_to(DOWN * 2.5)
        self.play(FadeIn(note2), run_time=0.3)
        self.wait(2)
