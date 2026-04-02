"""
Figure 2: Tiled Addition — Load, Compute, Store
Shows a [128] vector split into 8 tiles. Zooms into one tile to show:
  1. DMA load lhs tile + rhs tile into local memory
  2. Element-wise add in local
  3. Result written back to output
"""
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from theme import parse_theme

from manim import *

C, THEME = parse_theme()


def make_row(n, x0, y0, w, h, fill, label_fn=None, stroke=C["fg3"]):
    g = VGroup()
    for i in range(n):
        r = Rectangle(width=w, height=h, fill_color=fill, fill_opacity=0.5,
                      stroke_color=stroke, stroke_width=1)
        r.move_to([x0 + i * w, y0, 0])
        if label_fn:
            t = Text(label_fn(i), font_size=9, color=C["fg"], font="Monospace").move_to(r)
            g.add(VGroup(r, t))
        else:
            g.add(r)
    return g


class TiledAdd(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text("Tiled Element-Wise Addition", font_size=30, color=C["fg"], font="Monospace")
        title.to_edge(UP, buff=0.3)
        self.add(title)

        # --- Top: full [128] vectors ---
        top_y = 2.3
        full_w = 0.32
        n_tiles = 8
        tile_size = 16
        x_start = -n_tiles * full_w / 2 + full_w / 2

        lhs_label = Text("lhs [128]", font_size=14, color=C["lhs_c"], font="Monospace")
        lhs_label.move_to([0, top_y + 0.4, 0])
        self.add(lhs_label)

        lhs_tiles = VGroup()
        for t in range(n_tiles):
            r = Rectangle(width=full_w - 0.02, height=0.35,
                          fill_color=C["lhs_c"] if t != 2 else C["lhs_c"],
                          fill_opacity=0.25 if t != 2 else 0.7,
                          stroke_color=C["lhs_c"], stroke_width=1.5 if t == 2 else 0.8)
            r.move_to([x_start + t * full_w, top_y, 0])
            lbl = Text(str(t), font_size=8, color=C["fg"], font="Monospace").move_to(r)
            lhs_tiles.add(VGroup(r, lbl))
        self.add(lhs_tiles)

        rhs_label = Text("rhs [128]", font_size=14, color=C["rhs_c"], font="Monospace")
        rhs_label.move_to([0, top_y - 0.7, 0])
        self.add(rhs_label)

        rhs_tiles = VGroup()
        for t in range(n_tiles):
            r = Rectangle(width=full_w - 0.02, height=0.35,
                          fill_color=C["rhs_c"] if t != 2 else C["rhs_c"],
                          fill_opacity=0.25 if t != 2 else 0.7,
                          stroke_color=C["rhs_c"], stroke_width=1.5 if t == 2 else 0.8)
            r.move_to([x_start + t * full_w, top_y - 1.05, 0])
            lbl = Text(str(t), font_size=8, color=C["fg"], font="Monospace").move_to(r)
            rhs_tiles.add(VGroup(r, lbl))
        self.add(rhs_tiles)

        # Bracket highlighting tile 2
        tile2_x = x_start + 2 * full_w
        hl = Text("tile = 2", font_size=12, color=C["yellow"], font="Monospace")
        hl.move_to([tile2_x, top_y + 0.75, 0])
        arr_hl = Arrow([tile2_x, top_y + 0.6, 0], [tile2_x, top_y + 0.22, 0],
                       buff=0, stroke_width=1.5, color=C["yellow"],
                       max_tip_length_to_length_ratio=0.2)
        self.add(hl, arr_hl)

        # --- Middle: zoomed tile in local memory ---
        mid_y = -0.2
        cell_w = 0.42
        n_cells = 6  # show 6 of 16 cells for space

        local_box = Rectangle(width=6.5, height=2.8, fill_color=C["green_dk"],
                              fill_opacity=0.1, stroke_color=C["green_dk"], stroke_width=1.5)
        local_box.move_to([0, mid_y - 0.2, 0])
        local_label = Text("Local Memory", font_size=16, color=C["green_dk"], font="Monospace")
        local_label.move_to([0, mid_y + 1.2, 0])
        self.add(local_box, local_label)

        # lhs tile in local
        lhs_local_label = Text("lhs_load.data", font_size=12, color=C["lhs_c"], font="Monospace")
        lhs_local_label.move_to([-2.5, mid_y + 0.7, 0])
        lhs_local = VGroup()
        for i in range(n_cells):
            r = Rectangle(width=cell_w, height=0.35, fill_color=C["lhs_c"],
                          fill_opacity=0.5, stroke_color=C["lhs_c"], stroke_width=1)
            r.move_to([-2.5 + (i - 2.5) * (cell_w + 0.02), mid_y + 0.35, 0])
            v = Text(f"a{32+i}", font_size=8, color=C["fg"], font="Monospace").move_to(r)
            lhs_local.add(VGroup(r, v))
        dots_l = Text("...", font_size=14, color=C["fg2"], font="Monospace")
        dots_l.next_to(lhs_local, RIGHT, buff=0.1)
        self.add(lhs_local_label, lhs_local, dots_l)

        # rhs tile in local
        rhs_local_label = Text("rhs_load.data", font_size=12, color=C["rhs_c"], font="Monospace")
        rhs_local_label.move_to([2.5, mid_y + 0.7, 0])
        rhs_local = VGroup()
        for i in range(n_cells):
            r = Rectangle(width=cell_w, height=0.35, fill_color=C["rhs_c"],
                          fill_opacity=0.5, stroke_color=C["rhs_c"], stroke_width=1)
            r.move_to([2.5 + (i - 2.5) * (cell_w + 0.02), mid_y + 0.35, 0])
            v = Text(f"b{32+i}", font_size=8, color=C["fg"], font="Monospace").move_to(r)
            rhs_local.add(VGroup(r, v))
        dots_r = Text("...", font_size=14, color=C["fg2"], font="Monospace")
        dots_r.next_to(rhs_local, RIGHT, buff=0.1)
        self.add(rhs_local_label, rhs_local, dots_r)

        # DMA arrows from top to local
        dma_arr_l = Arrow([tile2_x - 0.3, top_y - 0.25, 0],
                          [-2.5, mid_y + 0.95, 0],
                          buff=0.1, stroke_width=2, color=C["arrow_c"],
                          max_tip_length_to_length_ratio=0.08)
        dma_lbl_l = Text("dma.copy => local", font_size=10, color=C["arrow_c"], font="Monospace")
        dma_lbl_l.move_to([-3.2, top_y - 0.7, 0])

        dma_arr_r = Arrow([tile2_x + 0.3, top_y - 1.3, 0],
                          [2.5, mid_y + 0.95, 0],
                          buff=0.1, stroke_width=2, color=C["arrow_c"],
                          max_tip_length_to_length_ratio=0.08)
        dma_lbl_r = Text("dma.copy => local", font_size=10, color=C["arrow_c"], font="Monospace")
        dma_lbl_r.move_to([3.2, top_y - 1.7, 0])
        self.add(dma_arr_l, dma_lbl_l, dma_arr_r, dma_lbl_r)

        # + sign between local tiles
        plus = Text("+", font_size=28, color=C["fg"], font="Monospace")
        plus.move_to([0, mid_y + 0.35, 0])
        self.add(plus)

        # = result row
        eq = Text("=", font_size=24, color=C["fg"], font="Monospace")
        eq.move_to([0, mid_y - 0.3, 0])
        self.add(eq)

        result_cells = VGroup()
        for i in range(n_cells):
            r = Rectangle(width=cell_w, height=0.35, fill_color=C["out_c"],
                          fill_opacity=0.5, stroke_color=C["out_c"], stroke_width=1)
            r.move_to([(i - 2.5) * (cell_w + 0.02), mid_y - 0.7, 0])
            v = Text(f"a{32+i}+b{32+i}", font_size=6, color=C["fg"], font="Monospace").move_to(r)
            result_cells.add(VGroup(r, v))
        dots_o = Text("...", font_size=14, color=C["fg2"], font="Monospace")
        dots_o.next_to(result_cells, RIGHT, buff=0.1)
        res_label = Text("output.at(tile # i)", font_size=12, color=C["out_c"], font="Monospace")
        res_label.move_to([0, mid_y - 1.15, 0])
        self.add(result_cells, dots_o, res_label)

        # --- Bottom: output [128] ---
        bot_y = -2.8
        out_label = Text("output [128]", font_size=14, color=C["out_c"], font="Monospace")
        out_label.move_to([0, bot_y + 0.4, 0])
        self.add(out_label)

        out_tiles = VGroup()
        for t in range(n_tiles):
            r = Rectangle(width=full_w - 0.02, height=0.35,
                          fill_color=C["out_c"],
                          fill_opacity=0.25 if t != 2 else 0.7,
                          stroke_color=C["out_c"], stroke_width=1.5 if t == 2 else 0.8)
            r.move_to([x_start + t * full_w, bot_y, 0])
            lbl = Text(str(t), font_size=8, color=C["fg"], font="Monospace").move_to(r)
            out_tiles.add(VGroup(r, lbl))
        self.add(out_tiles)

        store_arr = Arrow([0, mid_y - 1.35, 0], [tile2_x, bot_y + 0.25, 0],
                          buff=0.1, stroke_width=2, color=C["out_c"],
                          max_tip_length_to_length_ratio=0.08)
        store_lbl = Text("write back", font_size=10, color=C["out_c"], font="Monospace")
        store_lbl.move_to([1.5, bot_y + 1.0, 0])
        self.add(store_arr, store_lbl)
