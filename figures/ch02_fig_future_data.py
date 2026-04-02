"""
Figure: Futures and .data — the indirection from DMA handle to actual tensor.
"""
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from theme import parse_theme

from manim import *

C, THEME = parse_theme()


class FutureData(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text("Futures and .data", font_size=26, color=C["fg"], font="Monospace")
        title.to_edge(UP, buff=0.3)
        self.add(title)

        # Timeline
        tl_y = 1.8
        tl = Arrow(LEFT * 5 + UP * tl_y, RIGHT * 5 + UP * tl_y,
                    buff=0, stroke_width=2, color=C["fg3"],
                    max_tip_length_to_length_ratio=0.02)
        tl_lbl = Text("time →", font_size=12, color=C["fg3"], font="Monospace")
        tl_lbl.move_to(RIGHT * 4.5 + UP * (tl_y + 0.25))
        self.add(tl, tl_lbl)

        # Phase 1: dma.copy issued
        p1_x = -3.0
        p1_dot = Dot([p1_x, tl_y, 0], radius=0.08, color=C["future_c"])
        p1_lbl = Text("dma.copy issued", font_size=11, color=C["future_c"], font="Monospace")
        p1_lbl.move_to([p1_x, tl_y + 0.35, 0])
        self.add(p1_dot, p1_lbl)

        # Transfer in progress
        transfer_bar = Rectangle(width=3.5, height=0.25, fill_color=C["arrow"],
                                  fill_opacity=0.3, stroke_color=C["arrow"], stroke_width=1)
        transfer_bar.move_to([-0.5, tl_y, 0])
        transfer_lbl = Text("hardware DMA in flight", font_size=10,
                             color=C["arrow"], font="Monospace")
        transfer_lbl.move_to([-0.5, tl_y - 0.35, 0])
        self.add(transfer_bar, transfer_lbl)

        # Phase 2: .data access
        p2_x = 2.5
        p2_dot = Dot([p2_x, tl_y, 0], radius=0.08, color=C["data_c"])
        p2_lbl = Text(".data access", font_size=11, color=C["data_c"], font="Monospace")
        p2_lbl.move_to([p2_x, tl_y + 0.35, 0])
        self.add(p2_dot, p2_lbl)

        # Future object
        future_box = Rectangle(width=4.5, height=2.5, fill_color=C["future_c"],
                                fill_opacity=0.08, stroke_color=C["future_c"], stroke_width=2)
        future_box.move_to(LEFT * 2.5 + DOWN * 0.8)
        future_title = Text("lhs_load  (DMA Future)", font_size=15,
                             color=C["future_c"], font="Monospace")
        future_title.move_to(future_box.get_top() + DOWN * 0.3)

        status_t = Text("status: in_flight → complete", font_size=11,
                          color=C["fg2"], font="Monospace")
        status_t.move_to(LEFT * 2.5 + DOWN * 0.5)

        data_t = Text(".data → spanned tensor in local mem", font_size=11,
                        color=C["data_c"], font="Monospace")
        data_t.move_to(LEFT * 2.5 + DOWN * 1.0)

        shape_t = Text("shape: [chunk_size]", font_size=11,
                         color=C["fg3"], font="Monospace")
        shape_t.move_to(LEFT * 2.5 + DOWN * 1.45)

        self.add(future_box, future_title, status_t, data_t, shape_t)

        # Data tensor (accessed via .data)
        data_box = Rectangle(width=3.5, height=2.0, fill_color=C["data_c"],
                              fill_opacity=0.1, stroke_color=C["data_c"], stroke_width=2)
        data_box.move_to(RIGHT * 3.0 + DOWN * 0.8)
        data_title = Text("lhs_load.data", font_size=15, color=C["data_c"], font="Monospace")
        data_title.move_to(data_box.get_top() + DOWN * 0.3)

        data_cells = VGroup()
        for i in range(6):
            r = Rectangle(width=0.4, height=0.35, fill_color=C["data_c"],
                          fill_opacity=0.4, stroke_color=C["data_c"], stroke_width=1)
            r.move_to(RIGHT * 3.0 + LEFT * 1.2 + RIGHT * i * 0.45 + DOWN * 0.7)
            v = Text(f"{i}", font_size=8, color=C["fg"], font="Monospace").move_to(r)
            data_cells.add(VGroup(r, v))
        dots = Text("...", font_size=12, color=C["fg2"], font="Monospace")
        dots.next_to(data_cells, RIGHT, buff=0.06)

        access_t = Text(".at(i)  → element at position i", font_size=11,
                          color=C["fg2"], font="Monospace")
        access_t.move_to(RIGHT * 3.0 + DOWN * 1.3)

        self.add(data_box, data_title, data_cells, dots, access_t)

        # Arrow from future to data
        arr = Arrow(future_box.get_right(), data_box.get_left(),
                     buff=0.15, stroke_width=3, color=C["data_c"],
                     max_tip_length_to_length_ratio=0.08)
        arr_lbl = Text(".data", font_size=14, color=C["data_c"], font="Monospace")
        arr_lbl.next_to(arr, UP, buff=0.1)
        self.add(arr, arr_lbl)

        # Code at bottom
        code = Text(
            "lhs_load.data.at(i)  // read element i from local memory",
            font_size=13, color=C["fg2"], font="Monospace"
        )
        code.move_to(DOWN * 2.8)
        self.add(code)

        note = Text(
            "sync now: .data valid immediately  |  async later: wait then .data",
            font_size=10, color=C["fg3"], font="Monospace"
        )
        note.move_to(DOWN * 3.3)
        self.add(note)
