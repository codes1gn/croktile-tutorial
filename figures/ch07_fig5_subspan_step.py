"""
Figure 5 (ch07): .subspan().step().at() — strided vs packed tiling.
Left: packed (step = tile size). Right: strided (step > tile size, gaps between tiles).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from theme import parse_theme
from manim import *

C, THEME = parse_theme()


class SubspanStep(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text("Packed tiling vs Strided tiling (.step)", font_size=18, color=C["fg"], font="Monospace")
        title.to_edge(UP, buff=0.3)
        self.add(title)

        cell = 0.28
        colors = [C["blue"], C["green"], C["orange"], C["purple"]]

        def draw_matrix(origin, label, tiles, total_r=8, total_c=8):
            """Draw a grid with colored tile regions."""
            for r in range(total_r):
                for c in range(total_c):
                    sq = Square(side_length=cell, fill_color=C["fill"], fill_opacity=0.15,
                                stroke_color=C["stroke"], stroke_width=0.6)
                    sq.move_to(origin + RIGHT * c * cell + DOWN * r * cell)
                    self.add(sq)

            for idx, (r0, c0, rh, ch) in enumerate(tiles):
                col = colors[idx % len(colors)]
                rect = Rectangle(width=ch * cell, height=rh * cell,
                                  fill_color=col, fill_opacity=0.3,
                                  stroke_color=col, stroke_width=1.5)
                rect.move_to(origin + RIGHT * (c0 * cell + (ch - 1) * cell / 2) +
                             DOWN * (r0 * cell + (rh - 1) * cell / 2))
                self.add(rect)
                t = Text(f"({idx // 2},{idx % 2})", font_size=12, color=col, font="Monospace")
                t.move_to(rect)
                self.add(t)

            lbl = Text(label, font_size=12, color=C["fg2"], font="Monospace")
            lbl.move_to(origin + RIGHT * (total_c * cell / 2 - cell / 2) + UP * 0.5)
            self.add(lbl)

        # Left: packed (step = tile size = 4)
        left_o = LEFT * 4.5 + UP * 1.0
        packed_tiles = [(0, 0, 4, 4), (0, 4, 4, 4), (4, 0, 4, 4), (4, 4, 4, 4)]
        draw_matrix(left_o, "Packed: step = tile size", packed_tiles)

        left_code = Text(".subspan(4,4).at(i,j)", font_size=13, color=C["fg3"], font="Monospace")
        left_code.move_to(left_o + DOWN * 2.8 + RIGHT * 0.8)
        self.add(left_code)

        # Right: strided (step = 6, tile = 3, so gaps exist)
        right_o = RIGHT * 1.0 + UP * 1.0
        strided_tiles = [(0, 0, 3, 3), (0, 5, 3, 3), (5, 0, 3, 3), (5, 5, 3, 3)]
        draw_matrix(right_o, "Strided: step > tile size", strided_tiles, total_r=10, total_c=10)

        right_code = Text(".subspan(3,3).step(5,5).at(i,j)", font_size=13, color=C["fg3"], font="Monospace")
        right_code.move_to(right_o + DOWN * 3.5 + RIGHT * 1.0)
        self.add(right_code)

        # Gap annotation
        gap_label = Text("gap", font_size=12, color=C["red"], font="Monospace")
        gap_label.move_to(right_o + RIGHT * 3 * cell + DOWN * 1.2 * cell)
        self.add(gap_label)
