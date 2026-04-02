"""
Animation 2: Tiled Addition — step by step.
1. Show full vector, split into tiles
2. Select a tile
3. DMA load lhs and rhs tiles to local
4. Element-wise addition in local
5. Write result back to output
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from theme import parse_theme
from manim import *

C, THEME = parse_theme()


class TiledAddAnim(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text("Tiled Addition: Step by Step", font_size=28,
                      color=C["fg"], font="Monospace")
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.6)

        n_tiles = 8
        cw = 0.55
        ox = -n_tiles * cw / 2 + cw / 2

        # Step 1: show vectors
        step1 = Text("Step 1: Two input vectors, split into 8 tiles",
                      font_size=16, color=C["fg2"], font="Monospace")
        step1.move_to(UP * 2.5)
        self.play(FadeIn(step1), run_time=0.3)

        lhs_lbl = Text("lhs [128]", font_size=14, color=C["lhs_c"], font="Monospace")
        lhs_lbl.move_to(LEFT * 4 + UP * 1.6)
        rhs_lbl = Text("rhs [128]", font_size=14, color=C["rhs_c"], font="Monospace")
        rhs_lbl.move_to(LEFT * 4 + UP * 0.5)

        lhs_tiles = VGroup()
        rhs_tiles = VGroup()
        for i in range(n_tiles):
            lr = Rectangle(width=cw - 0.04, height=0.4, fill_color=C["lhs_c"],
                           fill_opacity=0.3, stroke_color=C["lhs_c"], stroke_width=1)
            lr.move_to([ox + i * cw, 1.6, 0])
            lt = Text(str(i), font_size=10, color=C["fg"], font="Monospace").move_to(lr)
            lhs_tiles.add(VGroup(lr, lt))

            rr = Rectangle(width=cw - 0.04, height=0.4, fill_color=C["rhs_c"],
                           fill_opacity=0.3, stroke_color=C["rhs_c"], stroke_width=1)
            rr.move_to([ox + i * cw, 0.5, 0])
            rt = Text(str(i), font_size=10, color=C["fg"], font="Monospace").move_to(rr)
            rhs_tiles.add(VGroup(rr, rt))

        self.play(FadeIn(lhs_lbl), FadeIn(rhs_lbl),
                  *[FadeIn(t) for t in lhs_tiles],
                  *[FadeIn(t) for t in rhs_tiles], run_time=0.6)
        self.wait(0.5)

        # Step 2: highlight tile 2
        step2 = Text("Step 2: Select tile = 2", font_size=16, color=C["fg2"], font="Monospace")
        self.play(
            FadeOut(step1), FadeIn(step2.move_to(UP * 2.5)),
            lhs_tiles[2][0].animate.set_fill(C["lhs_c"], 0.8).set_stroke(C["yellow"], 2),
            rhs_tiles[2][0].animate.set_fill(C["rhs_c"], 0.8).set_stroke(C["yellow"], 2),
            run_time=0.5
        )
        self.wait(0.5)

        # Step 3: DMA load into local
        step3 = Text("Step 3: dma.copy both tiles => local",
                      font_size=16, color=C["fg2"], font="Monospace")

        local_box = Rectangle(width=5, height=1.8, fill_color=C["green_dk"],
                              fill_opacity=0.08, stroke_color=C["green_dk"], stroke_width=1.5)
        local_box.move_to(DOWN * 1.3)
        local_lbl = Text("Local Memory", font_size=14, color=C["green_dk"], font="Monospace")
        local_lbl.move_to(DOWN * 0.25)

        lhs_local = VGroup()
        rhs_local = VGroup()
        ncells = 4
        for i in range(ncells):
            lr = Rectangle(width=0.5, height=0.35, fill_color=C["lhs_c"],
                           fill_opacity=0.5, stroke_color=C["lhs_c"], stroke_width=1)
            lr.move_to([-1.5 + i * 0.55, -1.0, 0])
            lt = Text(f"a{32+i}", font_size=8, color=C["fg"], font="Monospace").move_to(lr)
            lhs_local.add(VGroup(lr, lt))

            rr = Rectangle(width=0.5, height=0.35, fill_color=C["rhs_c"],
                           fill_opacity=0.5, stroke_color=C["rhs_c"], stroke_width=1)
            rr.move_to([1.0 + i * 0.55, -1.0, 0])
            rt = Text(f"b{32+i}", font_size=8, color=C["fg"], font="Monospace").move_to(rr)
            rhs_local.add(VGroup(rr, rt))

        dots_l = Text("...", font_size=12, color=C["fg2"], font="Monospace")
        dots_l.next_to(lhs_local, RIGHT, buff=0.08)
        dots_r = Text("...", font_size=12, color=C["fg2"], font="Monospace")
        dots_r.next_to(rhs_local, RIGHT, buff=0.08)

        arr_l = Arrow(lhs_tiles[2][0].get_bottom(), [-1.0, -0.6, 0],
                      buff=0.1, stroke_width=2, color=C["arrow_c"],
                      max_tip_length_to_length_ratio=0.1)
        arr_r = Arrow(rhs_tiles[2][0].get_bottom(), [1.5, -0.6, 0],
                      buff=0.1, stroke_width=2, color=C["arrow_c"],
                      max_tip_length_to_length_ratio=0.1)

        self.play(
            FadeOut(step2), FadeIn(step3.move_to(UP * 2.5)),
            FadeIn(local_box), FadeIn(local_lbl),
            GrowArrow(arr_l), GrowArrow(arr_r),
            run_time=0.5
        )
        self.play(
            *[FadeIn(c) for c in lhs_local],
            *[FadeIn(c) for c in rhs_local],
            FadeIn(dots_l), FadeIn(dots_r),
            run_time=0.5
        )
        self.wait(0.5)

        # Step 4: Element-wise add
        step4 = Text("Step 4: Element-wise add in local",
                      font_size=16, color=C["fg2"], font="Monospace")

        plus = Text("+", font_size=24, color=C["fg"], font="Monospace")
        plus.move_to([0, -1.0, 0])
        eq = Text("=", font_size=20, color=C["fg"], font="Monospace")
        eq.move_to([0, -1.7, 0])

        result = VGroup()
        for i in range(ncells):
            rr = Rectangle(width=0.55, height=0.35, fill_color=C["out_c"],
                           fill_opacity=0.5, stroke_color=C["out_c"], stroke_width=1)
            rr.move_to([-0.9 + i * 0.6, -2.0, 0])
            rt = Text(f"a+b", font_size=8, color=C["fg"], font="Monospace").move_to(rr)
            result.add(VGroup(rr, rt))

        dots_o = Text("...", font_size=12, color=C["fg2"], font="Monospace")
        dots_o.next_to(result, RIGHT, buff=0.08)

        self.play(
            FadeOut(step3), FadeIn(step4.move_to(UP * 2.5)),
            FadeIn(plus), run_time=0.3
        )
        self.play(
            FadeIn(eq),
            *[GrowFromCenter(c) for c in result],
            FadeIn(dots_o),
            run_time=0.5
        )
        self.wait(0.5)

        # Step 5: Write back
        step5 = Text("Step 5: Write result to output[tile]",
                      font_size=16, color=C["fg2"], font="Monospace")

        out_lbl = Text("output [128]", font_size=14, color=C["out_c"], font="Monospace")
        out_lbl.move_to(LEFT * 4 + DOWN * 3.0)

        out_tiles = VGroup()
        for i in range(n_tiles):
            rr = Rectangle(width=cw - 0.04, height=0.4,
                           fill_color=C["out_c"],
                           fill_opacity=0.15 if i != 2 else 0.7,
                           stroke_color=C["out_c"], stroke_width=1)
            rr.move_to([ox + i * cw, -3.0, 0])
            rt = Text(str(i), font_size=10, color=C["fg"], font="Monospace").move_to(rr)
            out_tiles.add(VGroup(rr, rt))

        store_arr = Arrow([0, -2.3, 0], [ox + 2 * cw, -2.7, 0],
                          buff=0.1, stroke_width=2, color=C["out_c"],
                          max_tip_length_to_length_ratio=0.1)

        self.play(
            FadeOut(step4), FadeIn(step5.move_to(UP * 2.5)),
            FadeIn(out_lbl),
            *[FadeIn(t) for t in out_tiles],
            GrowArrow(store_arr),
            run_time=0.6
        )
        self.wait(0.3)

        done = Text("Repeat for all 8 tiles", font_size=18,
                     color=C["yellow"], font="Monospace")
        done.move_to(DOWN * 3.6)
        self.play(FadeOut(step5), FadeIn(done), run_time=0.4)
        self.wait(2)
