"""
Figure 3: chunkat Semantics — 2D Tensor Slicing
Shows a [64, 128] tensor being sliced by chunkat(tr, tc) with parallel {tr, tc} by [4, 8].
Highlights one tile, shows how chunkat maps (tr, tc) to a rectangular sub-block.
"""
from manim import *

BG = "#1a1a2e"
GRID_C = "#37474F"
TILE_C = "#4CAF50"
HIGHLIGHT_C = "#FFEB3B"
DIM_C = "#78909C"
LABEL_C = "#B0BEC5"


class ChunkatSemantics(Scene):
    def construct(self):
        self.camera.background_color = BG

        title = Text("chunkat — 2D Tile Selection", font_size=28, color=WHITE, font="Monospace")
        title.to_edge(UP, buff=0.3)
        self.add(title)

        rows, cols = 4, 8
        cell_w, cell_h = 0.7, 0.65

        grid_origin_x = -cols * cell_w / 2 + cell_w / 2
        grid_origin_y = 1.5

        # Full tensor label
        tensor_label = Text("tensor [64, 128]", font_size=16, color=LABEL_C, font="Monospace")
        tensor_label.move_to([0, grid_origin_y + rows * cell_h / 2 + 0.35, 0])
        self.add(tensor_label)

        hl_r, hl_c = 1, 3  # highlighted tile

        grid = VGroup()
        for r in range(rows):
            for c in range(cols):
                is_hl = (r == hl_r and c == hl_c)
                fill = HIGHLIGHT_C if is_hl else GRID_C
                opacity = 0.7 if is_hl else 0.3
                stroke = HIGHLIGHT_C if is_hl else GREY_D
                sw = 2.5 if is_hl else 0.8

                rect = Rectangle(width=cell_w - 0.04, height=cell_h - 0.04,
                                 fill_color=fill, fill_opacity=opacity,
                                 stroke_color=stroke, stroke_width=sw)
                x = grid_origin_x + c * cell_w
                y = grid_origin_y - r * cell_h
                rect.move_to([x, y, 0])

                lbl = Text(f"({r},{c})", font_size=9, color=WHITE if is_hl else DIM_C,
                           font="Monospace").move_to(rect)
                grid.add(VGroup(rect, lbl))

        self.add(grid)

        # Row axis labels
        for r in range(rows):
            t = Text(f"tr={r}", font_size=11, color=DIM_C, font="Monospace")
            t.move_to([grid_origin_x - cell_w * 0.75, grid_origin_y - r * cell_h, 0])
            self.add(t)

        # Col axis labels
        for c in range(cols):
            t = Text(f"tc={c}", font_size=9, color=DIM_C, font="Monospace")
            t.move_to([grid_origin_x + c * cell_w, grid_origin_y + rows * cell_h / 2 + 0.05, 0])
            self.add(t)

        # Dimension annotations
        row_dim = Text("4 row-tiles  (64 / #tr = 64 / 4 = 16 rows each)",
                       font_size=11, color=DIM_C, font="Monospace")
        row_dim.move_to([0, grid_origin_y - rows * cell_h - 0.25, 0])

        col_dim = Text("8 col-tiles  (128 / #tc = 128 / 8 = 16 cols each)",
                       font_size=11, color=DIM_C, font="Monospace")
        col_dim.next_to(row_dim, DOWN, buff=0.15)
        self.add(row_dim, col_dim)

        # --- Zoomed tile ---
        zoom_x, zoom_y = 3.0, -2.0
        zoom_w, zoom_h = 2.2, 1.6

        zoom_rect = Rectangle(width=zoom_w, height=zoom_h,
                              fill_color=HIGHLIGHT_C, fill_opacity=0.15,
                              stroke_color=HIGHLIGHT_C, stroke_width=2)
        zoom_rect.move_to([zoom_x, zoom_y, 0])

        zoom_title = Text("chunkat(1, 3)", font_size=16, color=HIGHLIGHT_C, font="Monospace")
        zoom_title.move_to([zoom_x, zoom_y + zoom_h / 2 + 0.3, 0])

        # Mini grid inside zoomed tile
        mini_rows, mini_cols = 4, 4
        mini_cw, mini_ch = 0.4, 0.3
        mini_ox = zoom_x - (mini_cols - 1) * mini_cw / 2
        mini_oy = zoom_y + (mini_rows - 1) * mini_ch / 2 - 0.05

        mini_grid = VGroup()
        for mr in range(mini_rows):
            for mc in range(mini_cols):
                r = Rectangle(width=mini_cw - 0.02, height=mini_ch - 0.02,
                              fill_color=TILE_C, fill_opacity=0.4,
                              stroke_color=TILE_C, stroke_width=0.8)
                gx = mini_ox + mc * mini_cw
                gy = mini_oy - mr * mini_ch
                r.move_to([gx, gy, 0])

                real_r = hl_r * (64 // rows) + mr
                real_c = hl_c * (128 // cols) + mc
                v = Text(f"{real_r},{real_c}", font_size=6, color=WHITE, font="Monospace")
                v.move_to(r)
                mini_grid.add(VGroup(r, v))

        dots_r = Text("...", font_size=14, color=GREY_B, font="Monospace")
        dots_r.move_to([zoom_x + zoom_w / 2 - 0.15, zoom_y, 0])
        dots_b = Text("...", font_size=14, color=GREY_B, font="Monospace")
        dots_b.move_to([zoom_x, zoom_y - zoom_h / 2 + 0.15, 0])

        zoom_dim = Text("[16, 16] sub-tensor", font_size=12, color=TILE_C, font="Monospace")
        zoom_dim.move_to([zoom_x, zoom_y - zoom_h / 2 - 0.25, 0])

        self.add(zoom_rect, zoom_title, mini_grid, dots_r, dots_b, zoom_dim)

        # Arrow from highlighted cell to zoom
        src_x = grid_origin_x + hl_c * cell_w
        src_y = grid_origin_y - hl_r * cell_h
        arrow = Arrow([src_x + cell_w / 2, src_y - cell_h / 2, 0],
                      [zoom_x - zoom_w / 2, zoom_y + zoom_h / 2, 0],
                      buff=0.1, stroke_width=2, color=HIGHLIGHT_C,
                      max_tip_length_to_length_ratio=0.06)
        self.add(arrow)

        # Code snippet at bottom
        code = Text(
            'parallel {tr, tc} by [4, 8]\n'
            '  tile = lhs.chunkat(tr, tc)  // [16, 16] sub-block',
            font_size=13, color=GREY_A, font="Monospace"
        )
        code.move_to([-2.0, -2.8, 0])

        # Compose explanation
        compose = Text(
            "Index: output.at(tr # i, tc # j)\n"
            "       = output.at(tr*16 + i, tc*16 + j)",
            font_size=11, color=DIM_C, font="Monospace"
        )
        compose.move_to([-2.0, -3.6, 0])

        self.add(code, compose)
