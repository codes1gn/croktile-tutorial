"""
Figure: The # Extent Operator — #tile gives the number of tiles.
Shows how #tile is used to compute inner loop bounds.
"""
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from theme import parse_theme

from manim import *

C, THEME = parse_theme()


class ExtentOperator(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text("The # Extent Operator", font_size=26, color=C["fg"], font="Monospace")
        title.to_edge(UP, buff=0.3)
        self.add(title)

        # Tensor visualization
        vec_y = 1.8
        n_tiles = 8
        tile_size = 16
        total = 128
        cw = 0.55
        ox = -n_tiles * cw / 2 + cw / 2

        vec_lbl = Text(f"vector [{total}]", font_size=14, color=C["fg2"], font="Monospace")
        vec_lbl.move_to([0, vec_y - 0.4, 0])
        self.add(vec_lbl)

        tiles = VGroup()
        for t in range(n_tiles):
            r = Rectangle(width=cw - 0.04, height=0.45,
                          fill_color=C["orange"], fill_opacity=0.3,
                          stroke_color=C["orange"], stroke_width=1)
            r.move_to([ox + t * cw, vec_y, 0])
            lbl = Text(str(t), font_size=12, color=C["fg"], font="Monospace").move_to(r)
            tiles.add(VGroup(r, lbl))
        self.add(tiles)

        # Brace showing #tile = 8
        brace_top = Brace(tiles, UP, buff=0.05, color=C["extent_c"])
        brace_lbl = Text("#tile = 8  (number of tiles)", font_size=13,
                          color=C["extent_c"], font="Monospace")
        brace_lbl.next_to(brace_top, UP, buff=0.1)
        self.add(brace_top, brace_lbl)

        # Tile size note
        tile_size_lbl = Text(f"each tile: {total} / #tile = {total} / {n_tiles} = {tile_size} elements",
                              font_size=13, color=C["fg2"], font="Monospace")
        tile_size_lbl.move_to([0, vec_y - 0.8, 0])
        self.add(tile_size_lbl)

        # The two uses of #
        box_y = -1.0
        uses_box = Rectangle(width=10, height=2.8, fill_color=C["fill"],
                              fill_opacity=0.2, stroke_color=C["dim"], stroke_width=1)
        uses_box.move_to([0, box_y, 0])

        uses_title = Text("Two meanings of #", font_size=18, color=C["fg"], font="Monospace")
        uses_title.move_to([0, box_y + 1.1, 0])

        # Prefix use
        prefix_label = Text("Prefix:  #tile", font_size=16, color=C["extent_c"], font="Monospace")
        prefix_label.move_to([0, box_y + 0.45, 0])
        prefix_desc = Text('→ extent (count) = 8  |  "how many tiles?"', font_size=12,
                            color=C["fg2"], font="Monospace")
        prefix_desc.move_to([0, box_y + 0.05, 0])
        prefix_ex = Text("foreach i in [128 / #tile]  →  foreach i in [16]",
                          font_size=12, color=C["dim_c"], font="Monospace")
        prefix_ex.move_to([0, box_y - 0.35, 0])

        # Infix use
        infix_label = Text("Infix:  tile # i", font_size=16, color=C["blue"], font="Monospace")
        infix_label.move_to([0, box_y - 0.85, 0])
        infix_desc = Text('→ compose = tile*16+i  |  "element i within tile"', font_size=12,
                           color=C["fg2"], font="Monospace")
        infix_desc.move_to([0, box_y - 1.2, 0])

        self.add(uses_box, uses_title, prefix_label, prefix_desc, prefix_ex,
                 infix_label, infix_desc)

        # Key rule
        key = Text(
            "#name alone = extent  |  a # b = compose  — context disambiguates",
            font_size=12, color=C["fg3"], font="Monospace"
        )
        key.move_to(DOWN * 3.2)
        self.add(key)
