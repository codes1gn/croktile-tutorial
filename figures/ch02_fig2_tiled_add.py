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
            t = Text(label_fn(i), font_size=12, color=C["fg"], font="Monospace").move_to(r)
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

        # --- Top: full [128] vectors (side by side) ---
        top_y = 1.6
        full_w = 0.32
        n_tiles = 8
        tile_size = 16
        lhs_cx = -2.5
        rhs_cx = 2.5
        lhs_x_start = lhs_cx - n_tiles * full_w / 2 + full_w / 2
        rhs_x_start = rhs_cx - n_tiles * full_w / 2 + full_w / 2

        lhs_label = Text("lhs [128]", font_size=14, color=C["lhs_c"], font="Monospace")
        lhs_label.move_to([lhs_cx, top_y + 0.4, 0])
        self.add(lhs_label)

        lhs_tiles = VGroup()
        for t in range(n_tiles):
            r = Rectangle(width=full_w - 0.02, height=0.35,
                          fill_color=C["lhs_c"],
                          fill_opacity=0.25 if t != 2 else 0.7,
                          stroke_color=C["lhs_c"], stroke_width=1.5 if t == 2 else 0.8)
            r.move_to([lhs_x_start + t * full_w, top_y, 0])
            lbl = Text(str(t), font_size=12, color=C["fg"], font="Monospace").move_to(r)
            lhs_tiles.add(VGroup(r, lbl))
        self.add(lhs_tiles)

        rhs_label = Text("rhs [128]", font_size=14, color=C["rhs_c"], font="Monospace")
        rhs_label.move_to([rhs_cx, top_y + 0.4, 0])
        self.add(rhs_label)

        rhs_tiles = VGroup()
        for t in range(n_tiles):
            r = Rectangle(width=full_w - 0.02, height=0.35,
                          fill_color=C["rhs_c"],
                          fill_opacity=0.25 if t != 2 else 0.7,
                          stroke_color=C["rhs_c"], stroke_width=1.5 if t == 2 else 0.8)
            r.move_to([rhs_x_start + t * full_w, top_y, 0])
            lbl = Text(str(t), font_size=12, color=C["fg"], font="Monospace").move_to(r)
            rhs_tiles.add(VGroup(r, lbl))
        self.add(rhs_tiles)

        # Bracket highlighting tile 2
        lhs_tile2_x = lhs_x_start + 2 * full_w
        rhs_tile2_x = rhs_x_start + 2 * full_w
        hl_l = Text("tile = 2", font_size=12, color=C["yellow"], font="Monospace")
        hl_l.move_to([lhs_tile2_x, top_y + 0.75, 0])
        arr_hl_l = Arrow([lhs_tile2_x, top_y + 0.6, 0], [lhs_tile2_x, top_y + 0.22, 0],
                         buff=0, stroke_width=1.5, color=C["yellow"],
                         max_tip_length_to_length_ratio=0.2)
        hl_r = Text("tile = 2", font_size=12, color=C["yellow"], font="Monospace")
        hl_r.move_to([rhs_tile2_x, top_y + 0.75, 0])
        arr_hl_r = Arrow([rhs_tile2_x, top_y + 0.6, 0], [rhs_tile2_x, top_y + 0.22, 0],
                         buff=0, stroke_width=1.5, color=C["yellow"],
                         max_tip_length_to_length_ratio=0.2)
        self.add(hl_l, arr_hl_l, hl_r, arr_hl_r)

        # --- Middle: zoomed tile in local memory ---
        mid_y = -0.2
        cell_w = 0.50
        n_cells = 5

        local_box = Rectangle(width=6.8, height=1.2, fill_color=C["green_dk"],
                              fill_opacity=0.1, stroke_color=C["green_dk"], stroke_width=1.5)
        local_box.move_to([0, mid_y + 0.3, 0])
        local_label = Text("Local Memory", font_size=13, color=C["green_dk"], font="Monospace")
        local_label.next_to(local_box, UP, buff=0.05)
        self.add(local_box, local_label)

        # lhs tile in local
        lhs_cx = -1.8
        lhs_local_label = Text("lhs_local", font_size=11, color=C["lhs_c"], font="Monospace")
        lhs_local = VGroup()
        for i in range(n_cells):
            r = Rectangle(width=cell_w, height=0.35, fill_color=C["lhs_c"],
                          fill_opacity=0.5, stroke_color=C["lhs_c"], stroke_width=1)
            r.move_to([lhs_cx + (i - n_cells / 2 + 0.5) * (cell_w + 0.02), mid_y + 0.35, 0])
            v = Text(f"a{32+i}", font_size=11, color=C["fg"], font="Monospace").move_to(r)
            lhs_local.add(VGroup(r, v))
        dots_l = Text("···", font_size=14, color=C["fg2"], font="Monospace")
        dots_l.next_to(lhs_local, RIGHT, buff=0.06)
        lhs_local_label.next_to(lhs_local, UP, buff=0.1)
        self.add(lhs_local_label, lhs_local, dots_l)

        # rhs tile in local
        rhs_cx = 1.8
        rhs_local_label = Text("rhs_local", font_size=11, color=C["rhs_c"], font="Monospace")
        rhs_local = VGroup()
        for i in range(n_cells):
            r = Rectangle(width=cell_w, height=0.35, fill_color=C["rhs_c"],
                          fill_opacity=0.5, stroke_color=C["rhs_c"], stroke_width=1)
            r.move_to([rhs_cx + (i - n_cells / 2 + 0.5) * (cell_w + 0.02), mid_y + 0.35, 0])
            v = Text(f"b{32+i}", font_size=11, color=C["fg"], font="Monospace").move_to(r)
            rhs_local.add(VGroup(r, v))
        dots_r = Text("···", font_size=14, color=C["fg2"], font="Monospace")
        dots_r.next_to(rhs_local, RIGHT, buff=0.06)
        rhs_local_label.next_to(rhs_local, UP, buff=0.1)
        self.add(rhs_local_label, rhs_local, dots_r)

        # DMA arrows from top to local
        local_lhs_cx = -1.8
        local_rhs_cx = 1.8
        dma_arr_l = Arrow([lhs_cx, top_y - 0.22, 0],
                          [local_lhs_cx, mid_y + 0.85, 0],
                          buff=0.05, stroke_width=2, color=C["arrow_c"],
                          max_tip_length_to_length_ratio=0.12)
        dma_lbl_l = Text("dma.copy", font_size=11, color=C["arrow_c"], font="Monospace")
        dma_lbl_l.next_to(dma_arr_l, LEFT, buff=0.08)

        dma_arr_r = Arrow([rhs_cx, top_y - 0.22, 0],
                          [local_rhs_cx, mid_y + 0.85, 0],
                          buff=0.05, stroke_width=2, color=C["arrow_c"],
                          max_tip_length_to_length_ratio=0.12)
        dma_lbl_r = Text("dma.copy", font_size=11, color=C["arrow_c"], font="Monospace")
        dma_lbl_r.next_to(dma_arr_r, RIGHT, buff=0.08)
        self.add(dma_arr_l, dma_lbl_l, dma_arr_r, dma_lbl_r)

        # + sign between local tiles
        plus = Text("+", font_size=28, color=C["fg"], font="Monospace")
        plus.move_to([0, mid_y + 0.35, 0])
        self.add(plus)

        # = sign
        res_y = mid_y - 0.6
        eq = Text("=", font_size=24, color=C["fg"], font="Monospace")
        eq.move_to([0, res_y, 0])
        self.add(eq)

        # Zoomed result: c32-c36 (above output row)
        zoom_y = res_y - 0.5
        result_cells = VGroup()
        for i in range(n_cells):
            r = Rectangle(width=cell_w, height=0.35, fill_color=C["out_c"],
                          fill_opacity=0.5, stroke_color=C["out_c"], stroke_width=1)
            r.move_to([(i - n_cells / 2 + 0.5) * (cell_w + 0.02), zoom_y, 0])
            v = Text(f"c{32+i}", font_size=11, color=C["fg"], font="Monospace").move_to(r)
            result_cells.add(VGroup(r, v))
        dots_o = Text("...  (16 elements)", font_size=11, color=C["fg2"], font="Monospace")
        dots_o.next_to(result_cells, RIGHT, buff=0.08)
        res_formula = Text("cᵢ = aᵢ + bᵢ", font_size=12, color=C["out_c"], font="Monospace")
        res_formula.next_to(result_cells, LEFT, buff=0.3)
        self.add(result_cells, dots_o, res_formula)

        # --- Bottom: output [128] ---
        bot_y = -2.8
        out_x_start = -n_tiles * full_w / 2 + full_w / 2
        out_tile2_x = out_x_start + 2 * full_w
        out_label = Text("output [128]", font_size=14, color=C["out_c"], font="Monospace")
        out_label.move_to([0, bot_y - 0.4, 0])
        self.add(out_label)

        out_tiles = VGroup()
        for t in range(n_tiles):
            r = Rectangle(width=full_w - 0.02, height=0.35,
                          fill_color=C["out_c"],
                          fill_opacity=0.25 if t != 2 else 0.7,
                          stroke_color=C["out_c"], stroke_width=1.5 if t == 2 else 0.8)
            r.move_to([out_x_start + t * full_w, bot_y, 0])
            lbl = Text(str(t), font_size=12, color=C["fg"], font="Monospace").move_to(r)
            out_tiles.add(VGroup(r, lbl))
        self.add(out_tiles)

        # Bracket lines from tile 2 edges to c32/c36 edges
        tile2_rect = out_tiles[2][0]
        t2_left = tile2_rect.get_left()[0]
        t2_right = tile2_rect.get_right()[0]
        t2_top = tile2_rect.get_top()[1]
        c32_left = result_cells[0][0].get_left()[0]
        c36_right = result_cells[-1][0].get_right()[0]
        c_bottom = result_cells[0][0].get_bottom()[1]

        line_l = Line([t2_left, t2_top, 0], [c32_left, c_bottom, 0],
                      stroke_width=1.5, color=C["out_c"])
        line_r = Line([t2_right, t2_top, 0], [c36_right+0.3, c_bottom, 0],
                      stroke_width=1.5, color=C["out_c"])
        self.add(line_l, line_r)
