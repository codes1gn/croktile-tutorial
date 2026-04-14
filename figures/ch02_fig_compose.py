"""
Figure: The # Compose Operator — mapping tile + local offset to global index.
Shows tile=2, i=3 composing to global index 35 in a 128-element vector with 8 tiles of 16.
"""
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from theme import parse_theme

from manim import *

C, THEME = parse_theme()


class ComposeOperator(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text("The # Compose Operator", font_size=26, color=C["fg"], font="Monospace")
        title.to_edge(UP, buff=0.3)
        self.add(title)

        subtitle = Text("tile # i  =  tile × tile_size + i", font_size=16,
                          color=C["fg2"], font="Monospace")
        subtitle.next_to(title, DOWN, buff=0.2)
        self.add(subtitle)

        # Full vector at top
        n_tiles = 8
        tile_size = 16
        cw = 0.6
        ox = -n_tiles * cw / 2 + cw / 2
        vec_y = 1.5

        vec_label = Text("output [128]  — global view", font_size=14,
                          color=C["blue"], font="Monospace")
        vec_label.move_to([0, vec_y + 0.5, 0])
        self.add(vec_label)

        tiles = VGroup()
        hl_tile = 2
        for t in range(n_tiles):
            r = Rectangle(width=cw - 0.04, height=0.45,
                          fill_color=C["orange"] if t == hl_tile else C["fill"],
                          fill_opacity=0.6 if t == hl_tile else 0.3,
                          stroke_color=C["orange"] if t == hl_tile else C["dim"],
                          stroke_width=2 if t == hl_tile else 0.8)
            r.move_to([ox + t * cw, vec_y, 0])
            lbl = Text(str(t), font_size=12, color=C["fg"], font="Monospace").move_to(r)
            tiles.add(VGroup(r, lbl))
        self.add(tiles)

        # Zoomed tile 2
        zoom_y = 0.2
        zoom_cw = 0.35
        n_show = 8

        zoom_label = Text("tile = 2  (elements 32–47)", font_size=13,
                           color=C["orange"], font="Monospace")
        zoom_label.move_to([0, zoom_y + 0.55, 0])
        self.add(zoom_label)

        zoom_cells = VGroup()
        hl_i = 3
        for i in range(n_show):
            r = Rectangle(width=zoom_cw - 0.02, height=0.35,
                          fill_color=C["green"] if i == hl_i else C["fill"],
                          fill_opacity=0.7 if i == hl_i else 0.3,
                          stroke_color=C["green"] if i == hl_i else C["dim"],
                          stroke_width=2 if i == hl_i else 0.8)
            r.move_to([(i - n_show / 2 + 0.5) * (zoom_cw + 0.02), zoom_y, 0])
            global_idx = hl_tile * tile_size + i
            v = Text(str(global_idx), font_size=12, color=C["fg"], font="Monospace").move_to(r)
            zoom_cells.add(VGroup(r, v))

        dots_z = Text("... (16 elements total)", font_size=12, color=C["fg2"], font="Monospace")
        dots_z.next_to(zoom_cells, RIGHT, buff=0.1)
        self.add(zoom_cells, dots_z)

        # Arrow from tile 2 left edge to cell 32's left edge
        tile2_left_x = ox + hl_tile * cw - (cw - 0.04) / 2
        cell32_left_x = (0 - n_show / 2 + 0.5) * (zoom_cw + 0.02) - (zoom_cw - 0.02) / 2
        arr_zoom = Arrow([tile2_left_x, vec_y - 0.25, 0],
                          [cell32_left_x, zoom_y + 0.2, 0],
                          buff=0.05, stroke_width=1.5, color=C["orange"],
                          max_tip_length_to_length_ratio=0.08)
        self.add(arr_zoom)

        # Local index labels
        for i in range(n_show):
            il = Text(f"i={i}", font_size=10, color=C["fg3"], font="Monospace")
            il.move_to([(i - n_show / 2 + 0.5) * (zoom_cw + 0.02), zoom_y - 0.28, 0])
            self.add(il)

        # Composition example
        example_y = -1.3
        ex_box = Rectangle(width=8, height=1.6, fill_color=C["fill"],
                            fill_opacity=0.3, stroke_color=C["dim"], stroke_width=1)
        ex_box.move_to([0, example_y, 0])

        ex_title = Text("Example: tile = 2, i = 3", font_size=15,
                          color=C["fg"], font="Monospace")
        ex_title.move_to([0, example_y + 0.5, 0])

        compose_eq = Text("tile # i  =  2 × 16 + 3  =  35", font_size=16,
                            color=C["blue"], font="Monospace")
        compose_eq.move_to([0, example_y, 0])

        meaning = Text("output.at(tile # i)  →  output.at(35)  →  element at position 35",
                         font_size=12, color=C["fg2"], font="Monospace")
        meaning.move_to([0, example_y - 0.45, 0])

        self.add(ex_box, ex_title, compose_eq, meaning)

        # Visual pointer to element 35 in zoom
        # ptr_arrow = Arrow([(hl_i - n_show / 2 + 0.5) * (zoom_cw + 0.02), zoom_y - 0.5, 0],
        #                    [(hl_i - n_show / 2 + 0.5) * (zoom_cw + 0.02), zoom_y - 0.22, 0],
        #                    buff=0, stroke_width=2, color=C["blue"],
        #                    max_tip_length_to_length_ratio=0.2)
        # ptr_lbl = Text("35", font_size=12, color=C["blue"], font="Monospace")
        # ptr_lbl.move_to([(hl_i - n_show / 2 + 0.5) * (zoom_cw + 0.02), zoom_y - 0.65, 0])
        # self.add(ptr_arrow, ptr_lbl)

        # Key insight
        key = Text(
            "outer # inner  →  the tile index goes LEFT, element offset goes RIGHT",
            font_size=12, color=C["fg3"], font="Monospace"
        )
        key.move_to(DOWN * 2.8)
        self.add(key)
