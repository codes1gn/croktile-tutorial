"""
Figure 6: DMA Matmul — block grid with K-loop and shared memory.
Shows the outer block grid, the K-loop loading tiles into shared,
and the inner thread grid computing within each block.

Dimensions (from code):
  lhs [128, 256]   → wider than tall
  rhs [256, 256]   → square
  lhs tile [16, 16] → square  (lhs.chunkat(px, tile_k))
  rhs tile [16, 16] → square  (rhs.chunkat(tile_k, py))
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from theme import parse_theme
from manim import *

C, THEME = parse_theme()


class DmaMatmul(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text("DMA Matmul: Block Grid + Shared Memory", font_size=20,
                      color=C["fg"], font="Monospace")
        title.to_edge(UP, buff=0.25)
        self.add(title)

        # ── Global matrices (left side) ──
        # lhs [128, 256]: 128 rows × 256 cols → wider than tall
        lhs_box = Rectangle(width=2.4, height=1.4, fill_color=C["lhs_c"],
                             fill_opacity=0.12, stroke_color=C["lhs_c"], stroke_width=1.5)
        lhs_box.move_to(LEFT * 5.3 + UP * 1.0)
        lhs_lbl = Text("lhs [128, 256]", font_size=10, color=C["lhs_c"],
                        font="Monospace")
        lhs_lbl.move_to(lhs_box.get_top() + DOWN * 0.18)
        self.add(lhs_box, lhs_lbl)

        # Highlighted tile inside lhs — [16, 16] square
        TILE_S = 0.45
        lhs_tile_g = Rectangle(width=TILE_S, height=TILE_S,
                                fill_color=C["lhs_c"], fill_opacity=0.35,
                                stroke_color=C["lhs_c"], stroke_width=1.5)
        lhs_tile_g.move_to(lhs_box.get_center() + LEFT * 0.4 + DOWN * 0.1)
        lhs_tile_t = Text("[16,16]", font_size=8, color=C["lhs_c"],
                           font="Monospace").move_to(lhs_tile_g)
        self.add(lhs_tile_g, lhs_tile_t)

        # rhs [256, 256]: square
        rhs_box = Rectangle(width=2.0, height=2.0, fill_color=C["rhs_c"],
                             fill_opacity=0.12, stroke_color=C["rhs_c"], stroke_width=1.5)
        rhs_box.move_to(LEFT * 5.3 + DOWN * 1.8)
        rhs_lbl = Text("rhs [256, 256]", font_size=10, color=C["rhs_c"],
                        font="Monospace")
        rhs_lbl.move_to(rhs_box.get_top() + DOWN * 0.18)
        self.add(rhs_box, rhs_lbl)

        # Highlighted tile inside rhs — [16, 16] square
        rhs_tile_g = Rectangle(width=TILE_S, height=TILE_S,
                                fill_color=C["rhs_c"], fill_opacity=0.35,
                                stroke_color=C["rhs_c"], stroke_width=1.5)
        rhs_tile_g.move_to(rhs_box.get_center() + LEFT * 0.3 + UP * 0.3)
        rhs_tile_t = Text("[16,16]", font_size=8, color=C["rhs_c"],
                           font="Monospace").move_to(rhs_tile_g)
        self.add(rhs_tile_g, rhs_tile_t)

        # ── Block detail (center/right) ──
        block_box = Rectangle(width=7.0, height=5.5, fill_color=C["orange"],
                               fill_opacity=0.03, stroke_color=C["orange"],
                               stroke_width=2)
        block_box.move_to(RIGHT * 1.5 + DOWN * 0.3)
        block_title = Text("Block (px, py) — 1 of 128 blocks", font_size=11,
                            color=C["orange"], font="Monospace")
        block_title.move_to(block_box.get_top() + DOWN * 0.2)
        self.add(block_box, block_title)

        # ── foreach tile_k container (inside block) ──
        foreach_box = Rectangle(width=6.2, height=4.5, fill_color=C["purple"],
                                 fill_opacity=0.04, stroke_color=C["purple"],
                                 stroke_width=1.5, stroke_opacity=0.7)
        foreach_box.move_to(RIGHT * 1.5 + DOWN * 0.55)
        foreach_lbl = Text("foreach tile_k in [16]", font_size=11,
                            color=C["purple"], font="Monospace")
        foreach_lbl.move_to(foreach_box.get_top() + DOWN * 0.2)
        self.add(foreach_box, foreach_lbl)

        # Loop-back arrow on foreach
        loop_arrow = CurvedArrow(
            foreach_box.get_right() + DOWN * 0.5,
            foreach_box.get_right() + UP * 0.5,
            angle=-1.2, color=C["purple"], stroke_width=1.5
        )
        loop_lbl = Text("×16", font_size=10, color=C["purple"],
                         font="Monospace")
        loop_lbl.next_to(loop_arrow, RIGHT, buff=0.08)
        self.add(loop_arrow, loop_lbl)

        # ── Shared Memory area (inside foreach) ──
        smem_box = Rectangle(width=5.0, height=1.3, fill_color=C["shared_c"],
                              fill_opacity=0.1, stroke_color=C["shared_c"],
                              stroke_width=1.5)
        smem_box.move_to(RIGHT * 1.3 + UP * 0.5)
        smem_lbl = Text("Shared Memory", font_size=10, color=C["shared_c"],
                         font="Monospace")
        smem_lbl.move_to(smem_box.get_top() + DOWN * 0.17)
        self.add(smem_box, smem_lbl)

        # lhs_load [16,16] — square tile in shared memory
        SMTILE = 0.7
        lhs_sm = Rectangle(width=SMTILE, height=SMTILE,
                            fill_color=C["lhs_c"], fill_opacity=0.3,
                            stroke_color=C["lhs_c"], stroke_width=1.5)
        lhs_sm.move_to(RIGHT * 0.3 + UP * 0.25)
        lhs_sm_t = Text("lhs_load\n[16,16]", font_size=8, color=C["lhs_c"],
                         font="Monospace").move_to(lhs_sm)
        self.add(lhs_sm, lhs_sm_t)

        # rhs_load [16,16] — square tile in shared memory
        rhs_sm = Rectangle(width=SMTILE, height=SMTILE,
                            fill_color=C["rhs_c"], fill_opacity=0.3,
                            stroke_color=C["rhs_c"], stroke_width=1.5)
        rhs_sm.move_to(RIGHT * 2.3 + UP * 0.25)
        rhs_sm_t = Text("rhs_load\n[16,16]", font_size=8, color=C["rhs_c"],
                         font="Monospace").move_to(rhs_sm)
        self.add(rhs_sm, rhs_sm_t)

        # ── DMA arrows: global tiles → shared tiles ──
        dma1 = Arrow(lhs_tile_g.get_right(), lhs_sm.get_left(), buff=0.08,
                     stroke_width=2, color=C["arrow"],
                     max_tip_length_to_length_ratio=0.05)
        dma2 = Arrow(rhs_tile_g.get_right(), rhs_sm.get_left(), buff=0.08,
                     stroke_width=2, color=C["arrow"],
                     max_tip_length_to_length_ratio=0.05)

        dma_lbl = Text("dma.copy => shared", font_size=9, color=C["arrow"],
                        font="Monospace")
        dma_lbl.next_to(dma1, UP, buff=0.06)
        self.add(dma1, dma2, dma_lbl)

        # ── Thread grid (inside foreach) ──
        tg_label = Text("256 threads (16×16) : thread", font_size=10,
                         color=C["green"], font="Monospace")
        tg_label.move_to(RIGHT * 1.3 + DOWN * 0.7)
        self.add(tg_label)

        tg_origin = RIGHT * 0.0 + DOWN * 1.15
        for r in range(4):
            for c_ in range(4):
                rect = Rectangle(width=0.38, height=0.3,
                                 fill_color=C["green"], fill_opacity=0.15,
                                 stroke_color=C["green"], stroke_width=0.5)
                rect.move_to(tg_origin + RIGHT * c_ * 0.42 + DOWN * r * 0.34)
                self.add(rect)

        tg_note = Text("(4×4 of 16×16)", font_size=9,
                        color=C["dim"], font="Monospace")
        tg_note.move_to(RIGHT * 3.0 + DOWN * 1.5)
        self.add(tg_note)

        # ── Compute annotation (inside foreach) ──
        compute = Text("output.at(px#qx, py#qy)\n  += lhs_load.data × rhs_load.data",
                        font_size=10, color=C["fg2"], font="Monospace")
        compute.move_to(RIGHT * 1.3 + DOWN * 2.55)
        self.add(compute)

        # ── Output in global memory (left side, below rhs) ──
        out_box = Rectangle(width=2.4, height=0.7, fill_color=C["out_c"],
                             fill_opacity=0.12, stroke_color=C["out_c"], stroke_width=1.5)
        out_box.move_to(LEFT * 5.3 + DOWN * 3.4)
        out_lbl = Text("output [128, 256]", font_size=10, color=C["out_c"],
                        font="Monospace").move_to(out_box)
        out_sublbl = Text("(global memory)", font_size=8, color=C["out_c"],
                           font="Monospace")
        out_sublbl.next_to(out_box, DOWN, buff=0.05)
        self.add(out_box, out_lbl, out_sublbl)

        # Write-back arrow from block to output
        wb_arr = Arrow(block_box.get_left() + DOWN * 1.5,
                       out_box.get_right(), buff=0.08,
                       stroke_width=2, color=C["out_c"],
                       max_tip_length_to_length_ratio=0.05)
        wb_lbl = Text("write back", font_size=9, color=C["out_c"],
                       font="Monospace")
        wb_lbl.next_to(wb_arr, DOWN, buff=0.05)
        self.add(wb_arr, wb_lbl)
