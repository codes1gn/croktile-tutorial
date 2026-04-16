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

        data_label = Text("actual data (M=6, K=8)", font_size=12, color=C["blue"], font="Monospace")
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
        tile_label.next_to(tile_rect, RIGHT - DOWN * 0.5, buff=0.35)
        tile_label.shift(UP * 0.55)
        self.add(tile_label)
        tile_arrow = Arrow(
            tile_rect.get_right() + UP * 0.75,
            tile_label.get_left() + DOWN * 0.08,
            buff=0.04,
            stroke_width=2.0,
            color=C["orange"],
            max_tip_length_to_length_ratio=0.16,
        )
        self.add(tile_arrow)

        # Edge boundary line
        edge_v = DashedLine(
            # True boundary sits between valid col=7 and out-of-bounds col=8.
            origin + RIGHT * (data_cols - 0.5) * cell + UP * 0.2,
            origin + RIGHT * (data_cols - 0.5) * cell + DOWN * (data_rows + 2) * cell,
            color=C["dim"], stroke_width=1.5, dash_length=0.06)
        edge_h = DashedLine(
            # True boundary sits between valid row=5 and out-of-bounds row=6.
            origin + DOWN * (data_rows - 0.5) * cell + LEFT * 0.2,
            origin + DOWN * (data_rows - 0.5) * cell + RIGHT * (data_cols + 3) * cell,
            color=C["dim"], stroke_width=1.5, dash_length=0.06)
        self.add(edge_v, edge_h)

        edge_label = Text("tensor boundary", font_size=11, color=C["dim"], font="Monospace")
        edge_label.move_to(
            origin + RIGHT * (data_cols + 5.7) * cell + DOWN * (data_rows + 3) * cell
        )
        self.add(edge_label)
        edge_arrow_v = Arrow(
            edge_h.get_center() + RIGHT * 2.1,
            edge_label.get_top() + LEFT * 0.7,
            buff=0.04,
            stroke_width=2.4,
            color=C["fg2"],
            max_tip_length_to_length_ratio=0.1,
        )
        edge_arrow_h = Arrow(
            edge_h.get_center() + RIGHT * 0.75 + DOWN * 0.95,
            edge_label.get_left() + DOWN * 0.01,
            buff=0.04,
            stroke_width=2.4,
            color=C["fg2"],
            max_tip_length_to_length_ratio=0.07,
        )
        self.add(edge_arrow_v, edge_arrow_h)

        # Code at bottom
        code = Text(
            "tma.copy src.subspan(4,4).at(1,1).zfill => shared",
            font_size=12, color=C["fg3"], font="Monospace")
        code.to_edge(DOWN, buff=0.5)
        self.add(code)

        note = Text("out-of-bounds elements written as zero -- MMA stays uniform", font_size=13, color=C["fg3"], font="Monospace")
        note.to_edge(DOWN, buff=0.25)
        self.add(note)
