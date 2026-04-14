"""
Figure 5: Scalar Matmul — output tiling grid.
Shows [128,256] output divided into a (p=16, q=64) tile grid,
highlighting how each (p,q) pair owns a small tile of the output.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from theme import parse_theme
from manim import *

C, THEME = parse_theme()


class ScalarMatmul(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text("Scalar Matmul: Output Tiling", font_size=22,
                      color=C["fg"], font="Monospace")
        title.to_edge(UP, buff=0.3)
        self.add(title)

        # Output matrix representation
        out_label = Text("output [128, 256]", font_size=14, color=C["out_c"],
                          font="Monospace")
        out_label.move_to(UP * 2.2)
        self.add(out_label)

        grid_rows = 4
        grid_cols = 8
        cell_w = 0.85
        cell_h = 0.85
        origin = LEFT * (grid_cols * cell_w / 2 - cell_w / 2) + UP * 1.1
        grid_left = origin[0] - cell_w / 2
        grid_right = origin[0] + (grid_cols - 1) * cell_w + cell_w / 2
        grid_bottom = origin[1] - (grid_rows - 1) * cell_h - cell_h / 2
        grid_center_y = origin[1] - (grid_rows - 1) * cell_h / 2

        for r in range(grid_rows):
            for c_ in range(grid_cols):
                is_highlight = (r == 1 and c_ == 3)
                fill = C["out_c"] if is_highlight else C["fill"]
                opacity = 0.5 if is_highlight else 0.3
                stroke = C["out_c"] if is_highlight else C["stroke"]

                rect = Rectangle(width=cell_w, height=cell_h,
                                 fill_color=fill, fill_opacity=opacity,
                                 stroke_color=stroke, stroke_width=1)
                rect.move_to(origin + RIGHT * c_ * cell_w + DOWN * r * cell_h)

                if is_highlight:
                    lbl = Text(f"(p=1,q=3)\n8×4 tile", font_size=10,
                               color=C["fg"], font="Monospace")
                else:
                    lbl = Text(f"({r},{c_})", font_size=10, color=C["fg3"],
                               font="Monospace")
                lbl.move_to(rect)
                self.add(rect, lbl)

        right_dots = Text("...", font_size=18, color=C["fg2"], font="Monospace")
        right_dots.move_to([grid_right + 0.25, grid_center_y, 0])
        bottom_dots = Text("⋮", font_size=20, color=C["fg2"], font="Monospace")
        bottom_dots.move_to([0, grid_bottom - 0.16, 0])
        self.add(right_dots, bottom_dots)

        # Row/col annotations
        p_lbl = Text("p: 0..15 (16 row-tiles)", font_size=12, color=C["fg2"],
                      font="Monospace")
        p_lbl.move_to([grid_left - 0.55, -0.3, 0])
        p_lbl.rotate(PI / 2)

        q_lbl = Text("q: 0..63 (64 col-tiles)", font_size=12, color=C["fg2"],
                      font="Monospace")
        q_lbl.move_to([0, grid_bottom - 0.42, 0])
        self.add(p_lbl, q_lbl)

        # Dimensions
        dim_m = Text("128 rows", font_size=12, color=C["fg3"], font="Monospace")
        dim_m.move_to([grid_left - 0.85, 0.5, 0])
        dim_m.rotate(PI / 2)
        dim_n = Text("256 cols", font_size=12, color=C["fg3"], font="Monospace")
        dim_n.move_to(UP * 1.7)
        self.add(dim_m, dim_n)

        # Note about grid (only showing 4x8 of the 16x64 grid)
        note = Text("(showing 4×8 of the 16×64 tile grid)", font_size=12,
                     color=C["dim"], font="Monospace")
        note.to_edge(DOWN, buff=0.4)
        self.add(note)

        # Code snippet
        code = Text(
            "parallel {p, q} by [16, 64]\n"
            "  foreach {m,n,k} in [8, 4, 256]\n"
            "    output.at(p#m, q#n) += ...",
            font_size=12, color=C["fg2"], font="Monospace"
        )
        code.move_to(DOWN * 3.0)
        self.add(code)
