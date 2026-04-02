"""
Figure 6 (ch07): .zfill — partial tile zero-fill at boundary.
Shows a matrix where the last tile extends past the edge,
and .zfill pads out-of-bounds elements with zero.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from theme import parse_theme
from manim import *

C, THEME = parse_theme()


class ZFill(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text(".zfill: zero-padding partial tiles at boundary", font_size=18, color=C["fg"], font="Monospace")
        title.to_edge(UP, buff=0.3)
        self.add(title)

        cell = 0.36
        origin = LEFT * 2.5 + UP * 1.0

        # Draw a 6x8 data region (actual data)
        data_rows, data_cols = 6, 8
        for r in range(data_rows):
            for c in range(data_cols):
                sq = Square(side_length=cell, fill_color=C["blue"], fill_opacity=0.2,
                            stroke_color=C["blue"], stroke_width=0.8)
                sq.move_to(origin + RIGHT * c * cell + DOWN * r * cell)
                self.add(sq)

        data_label = Text("actual data (M=6, K=8)", font_size=11, color=C["blue"], font="Monospace")
        data_label.move_to(origin + RIGHT * 3.5 * cell + UP * 0.5)
        self.add(data_label)

        # Draw tile boundary at (4,6) with size 4x4, extending past edge
        tile_r0, tile_c0, tile_h, tile_w = 4, 6, 4, 4

        # Out-of-bounds cells (zeros)
        for r in range(tile_h):
            for c in range(tile_w):
                gr, gc = tile_r0 + r, tile_c0 + c
                if gr >= data_rows or gc >= data_cols:
                    sq = Square(side_length=cell, fill_color=C["red"], fill_opacity=0.15,
                                stroke_color=C["red"], stroke_width=0.8)
                    sq.move_to(origin + RIGHT * gc * cell + DOWN * gr * cell)
                    self.add(sq)
                    z = Text("0", font_size=12, color=C["red"], font="Monospace")
                    z.move_to(sq)
                    self.add(z)

        # Highlight the tile
        tile_rect = Rectangle(width=tile_w * cell, height=tile_h * cell,
                               fill_opacity=0, stroke_color=C["orange"], stroke_width=2)
        tile_rect.move_to(origin + RIGHT * (tile_c0 * cell + (tile_w - 1) * cell / 2) +
                          DOWN * (tile_r0 * cell + (tile_h - 1) * cell / 2))
        self.add(tile_rect)

        tile_label = Text("tile at(1,1) with .zfill", font_size=13, color=C["orange"], font="Monospace")
        tile_label.next_to(tile_rect, RIGHT, buff=0.15)
        self.add(tile_label)

        # Edge boundary line
        edge_v = DashedLine(
            origin + RIGHT * data_cols * cell + UP * 0.2,
            origin + RIGHT * data_cols * cell + DOWN * (data_rows + 2) * cell,
            color=C["dim"], stroke_width=1.5, dash_length=0.06)
        edge_h = DashedLine(
            origin + DOWN * data_rows * cell + LEFT * 0.2,
            origin + DOWN * data_rows * cell + RIGHT * (data_cols + 3) * cell,
            color=C["dim"], stroke_width=1.5, dash_length=0.06)
        self.add(edge_v, edge_h)

        edge_label = Text("tensor boundary", font_size=12, color=C["dim"], font="Monospace")
        edge_label.move_to(origin + RIGHT * (data_cols + 1.5) * cell + DOWN * data_rows * cell)
        self.add(edge_label)

        # Code at bottom
        code = Text(
            "tma.copy src.subspan(4,4).at(1,1).zfill => shared",
            font_size=11, color=C["fg3"], font="Monospace")
        code.to_edge(DOWN, buff=0.5)
        self.add(code)

        note = Text("out-of-bounds elements written as zero -- MMA stays uniform", font_size=13, color=C["fg3"], font="Monospace")
        note.to_edge(DOWN, buff=0.25)
        self.add(note)
