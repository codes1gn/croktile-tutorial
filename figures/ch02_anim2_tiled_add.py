"""
Animation 2: Tiled Addition — step by step.
Layout matches ch02_fig2_tiled_add.py exactly:
  lhs/rhs side by side at top, local memory in middle, result row,
  output [128] at bottom with write-back arrow.
"""
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from theme import parse_theme
from manim import *

C, THEME = parse_theme()


class TiledAddAnim(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text("Tiled Element-Wise Addition", font_size=30,
                      color=C["fg"], font="Monospace")
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.6)

        # ── Coordinates from the static figure ──
        top_y = 1.6
        full_w = 0.32
        n_tiles = 8
        lhs_cx = -2.5
        rhs_cx = 2.5
        lhs_x_start = lhs_cx - n_tiles * full_w / 2 + full_w / 2
        rhs_x_start = rhs_cx - n_tiles * full_w / 2 + full_w / 2
        lhs_tile2_x = lhs_x_start + 2 * full_w
        rhs_tile2_x = rhs_x_start + 2 * full_w
        mid_y = -0.2
        cell_w = 0.50
        n_cells = 5
        res_y = mid_y - 0.6
        bot_y = -3.0

        # ── Step 1: Show input vectors ──
        step1 = Text("Step 1: Two input vectors, split into 8 tiles",
                      font_size=14, color=C["fg2"], font="Monospace")
        step1.move_to(UP * 2.95)

        lhs_label = Text("lhs [128]", font_size=14, color=C["lhs_c"],
                          font="Monospace")
        lhs_label.move_to([lhs_cx, top_y + 0.4, 0])
        rhs_label = Text("rhs [128]", font_size=14, color=C["rhs_c"],
                          font="Monospace")
        rhs_label.move_to([rhs_cx, top_y + 0.4, 0])

        lhs_tiles = VGroup()
        rhs_tiles = VGroup()
        for t in range(n_tiles):
            lr = Rectangle(width=full_w - 0.02, height=0.35,
                           fill_color=C["lhs_c"], fill_opacity=0.25,
                           stroke_color=C["lhs_c"], stroke_width=0.8)
            lr.move_to([lhs_x_start + t * full_w, top_y, 0])
            lt = Text(str(t), font_size=12, color=C["fg"],
                      font="Monospace").move_to(lr)
            lhs_tiles.add(VGroup(lr, lt))

            rr = Rectangle(width=full_w - 0.02, height=0.35,
                           fill_color=C["rhs_c"], fill_opacity=0.25,
                           stroke_color=C["rhs_c"], stroke_width=0.8)
            rr.move_to([rhs_x_start + t * full_w, top_y, 0])
            rt = Text(str(t), font_size=12, color=C["fg"],
                      font="Monospace").move_to(rr)
            rhs_tiles.add(VGroup(rr, rt))

        self.play(
            FadeIn(step1),
            FadeIn(lhs_label), FadeIn(rhs_label),
            *[FadeIn(t) for t in lhs_tiles],
            *[FadeIn(t) for t in rhs_tiles],
            run_time=0.6
        )
        self.wait(0.5)

        # ── Step 2: Highlight tile 2 ──
        step2 = Text("Step 2: Select tile = 2", font_size=14,
                      color=C["fg2"], font="Monospace").move_to(UP * 2.95)

        hl_l = Text("tile = 2", font_size=12, color=C["yellow"],
                     font="Monospace")
        hl_l.move_to([lhs_tile2_x, top_y + 0.75, 0])
        arr_hl_l = Arrow([lhs_tile2_x, top_y + 0.6, 0],
                         [lhs_tile2_x, top_y + 0.22, 0],
                         buff=0, stroke_width=1.5, color=C["yellow"],
                         max_tip_length_to_length_ratio=0.2)
        hl_r = Text("tile = 2", font_size=12, color=C["yellow"],
                     font="Monospace")
        hl_r.move_to([rhs_tile2_x, top_y + 0.75, 0])
        arr_hl_r = Arrow([rhs_tile2_x, top_y + 0.6, 0],
                         [rhs_tile2_x, top_y + 0.22, 0],
                         buff=0, stroke_width=1.5, color=C["yellow"],
                         max_tip_length_to_length_ratio=0.2)

        self.play(
            ReplacementTransform(step1, step2),
            lhs_tiles[2][0].animate.set_fill(C["lhs_c"], 0.7).set_stroke(
                C["lhs_c"], 1.5),
            rhs_tiles[2][0].animate.set_fill(C["rhs_c"], 0.7).set_stroke(
                C["rhs_c"], 1.5),
            FadeIn(hl_l), GrowArrow(arr_hl_l),
            FadeIn(hl_r), GrowArrow(arr_hl_r),
            run_time=0.5
        )
        self.wait(0.5)

        # ── Step 3: DMA copy to local memory ──
        step3 = Text("Step 3: dma.copy both tiles => local", font_size=14,
                      color=C["fg2"], font="Monospace").move_to(UP * 2.95)

        local_box = Rectangle(width=6.8, height=1.2, fill_color=C["green_dk"],
                               fill_opacity=0.1, stroke_color=C["green_dk"],
                               stroke_width=1.5)
        local_box.move_to([0, mid_y + 0.3, 0])
        local_label = Text("Local Memory", font_size=13,
                            color=C["green_dk"], font="Monospace")
        local_label.next_to(local_box, UP, buff=0.05)

        local_lhs_cx = -1.8
        local_rhs_cx = 1.8

        lhs_local_label = Text("lhs_local", font_size=11, color=C["lhs_c"],
                                font="Monospace")
        lhs_local = VGroup()
        for i in range(n_cells):
            r = Rectangle(width=cell_w, height=0.35, fill_color=C["lhs_c"],
                          fill_opacity=0.5, stroke_color=C["lhs_c"],
                          stroke_width=1)
            r.move_to([local_lhs_cx + (i - n_cells / 2 + 0.5) * (cell_w + 0.02),
                       mid_y + 0.35, 0])
            v = Text(f"a{32+i}", font_size=11, color=C["fg"],
                     font="Monospace").move_to(r)
            lhs_local.add(VGroup(r, v))
        dots_l = Text("···", font_size=14, color=C["fg2"],
                       font="Monospace")
        dots_l.next_to(lhs_local, RIGHT, buff=0.08)
        lhs_local_label.next_to(lhs_local, UP, buff=0.1)

        rhs_local_label = Text("rhs_local", font_size=11, color=C["rhs_c"],
                                font="Monospace")
        rhs_local = VGroup()
        for i in range(n_cells):
            r = Rectangle(width=cell_w, height=0.35, fill_color=C["rhs_c"],
                          fill_opacity=0.5, stroke_color=C["rhs_c"],
                          stroke_width=1)
            r.move_to([local_rhs_cx + (i - n_cells / 2 + 0.5) * (cell_w + 0.02),
                       mid_y + 0.35, 0])
            v = Text(f"b{32+i}", font_size=11, color=C["fg"],
                     font="Monospace").move_to(r)
            rhs_local.add(VGroup(r, v))
        dots_r = Text("···", font_size=14, color=C["fg2"],
                       font="Monospace")
        dots_r.next_to(rhs_local, RIGHT, buff=0.08)
        rhs_local_label.next_to(rhs_local, UP, buff=0.1)

        plus = Text("+", font_size=28, color=C["fg"], font="Monospace")
        plus.move_to([0, mid_y + 0.35, 0])

        dma_arr_l = Arrow([lhs_cx, top_y - 0.22, 0],
                          [local_lhs_cx, mid_y + 0.85, 0],
                          buff=0.05, stroke_width=2, color=C["arrow_c"],
                          max_tip_length_to_length_ratio=0.12)
        dma_lbl_l = Text("dma.copy", font_size=11, color=C["arrow_c"],
                          font="Monospace")
        dma_lbl_l.next_to(dma_arr_l, LEFT, buff=0.08)

        dma_arr_r = Arrow([rhs_cx, top_y - 0.22, 0],
                          [local_rhs_cx, mid_y + 0.85, 0],
                          buff=0.05, stroke_width=2, color=C["arrow_c"],
                          max_tip_length_to_length_ratio=0.12)
        dma_lbl_r = Text("dma.copy", font_size=11, color=C["arrow_c"],
                          font="Monospace")
        dma_lbl_r.next_to(dma_arr_r, RIGHT, buff=0.08)

        self.play(
            ReplacementTransform(step2, step3),
            FadeIn(local_box), FadeIn(local_label),
            GrowArrow(dma_arr_l), FadeIn(dma_lbl_l),
            GrowArrow(dma_arr_r), FadeIn(dma_lbl_r),
            run_time=0.5
        )
        self.play(
            *[FadeIn(c) for c in lhs_local], FadeIn(lhs_local_label),
            FadeIn(dots_l),
            *[FadeIn(c) for c in rhs_local], FadeIn(rhs_local_label),
            FadeIn(dots_r),
            FadeIn(plus),
            run_time=0.5
        )
        self.wait(0.5)

        # ── Step 4: Element-wise add ──
        step4 = Text("Step 4: cᵢ = aᵢ + bᵢ  in local", font_size=14,
                      color=C["fg2"], font="Monospace").move_to(UP * 2.95)

        eq = Text("=", font_size=24, color=C["fg"], font="Monospace")
        eq.move_to([0, res_y, 0])

        result_cells = VGroup()
        for i in range(n_cells):
            r = Rectangle(width=cell_w, height=0.35, fill_color=C["out_c"],
                          fill_opacity=0.5, stroke_color=C["out_c"],
                          stroke_width=1)
            r.move_to([(i - n_cells / 2 + 0.5) * (cell_w + 0.02),
                       res_y - 0.4, 0])
            v = Text(f"c{32+i}", font_size=11, color=C["fg"],
                     font="Monospace").move_to(r)
            result_cells.add(VGroup(r, v))

        dots_o = Text("···", font_size=14, color=C["fg2"],
                       font="Monospace")
        dots_o.next_to(result_cells, RIGHT, buff=0.08)

        res_formula = Text("cᵢ = aᵢ + bᵢ", font_size=12, color=C["out_c"],
                            font="Monospace")
        res_formula.next_to(result_cells, LEFT, buff=0.3)

        res_label = Text("output tile (global)", font_size=11,
                          color=C["out_c"], font="Monospace")
        res_label.next_to(result_cells, DOWN, buff=0.1)

        self.play(
            ReplacementTransform(step3, step4),
            FadeIn(eq),
            run_time=0.3
        )
        self.play(
            *[GrowFromCenter(c) for c in result_cells],
            FadeIn(dots_o), FadeIn(res_formula), FadeIn(res_label),
            run_time=0.5
        )
        self.wait(0.5)

        # ── Step 5: Write back to output ──
        step5 = Text("Step 5: Write result to output[tile]", font_size=14,
                      color=C["fg2"], font="Monospace").move_to(UP * 2.95)

        out_x_start = -n_tiles * full_w / 2 + full_w / 2
        out_tile2_x = out_x_start + 2 * full_w

        out_label = Text("output [128]", font_size=14, color=C["out_c"],
                          font="Monospace")
        out_label.move_to([0, bot_y - 0.4, 0])

        out_tiles = VGroup()
        for t in range(n_tiles):
            r = Rectangle(width=full_w - 0.02, height=0.35,
                          fill_color=C["out_c"],
                          fill_opacity=0.15 if t != 2 else 0.7,
                          stroke_color=C["out_c"],
                          stroke_width=1.5 if t == 2 else 0.8)
            r.move_to([out_x_start + t * full_w, bot_y, 0])
            lbl = Text(str(t), font_size=12, color=C["fg"],
                       font="Monospace").move_to(r)
            out_tiles.add(VGroup(r, lbl))

        store_arr = Arrow([0, res_y - 0.75, 0],
                          [out_tile2_x, bot_y + 0.25, 0],
                          buff=0.1, stroke_width=2, color=C["out_c"],
                          max_tip_length_to_length_ratio=0.08)
        store_lbl = Text("write back", font_size=12, color=C["out_c"],
                          font="Monospace")
        store_lbl.next_to(store_arr, RIGHT, buff=0.08)

        self.play(
            ReplacementTransform(step4, step5),
            FadeIn(out_label),
            *[FadeIn(t) for t in out_tiles],
            GrowArrow(store_arr), FadeIn(store_lbl),
            run_time=0.6
        )
        self.wait(0.3)

        done = Text("Repeat for all 8 tiles", font_size=18,
                     color=C["yellow"], font="Monospace")
        done.move_to(DOWN * 3.65)
        self.play(FadeOut(step5), FadeIn(done), run_time=0.4)
        self.wait(2)
