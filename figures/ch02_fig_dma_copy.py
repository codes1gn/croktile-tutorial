"""
Figure: dma.copy — Bulk transfer between memory levels.
Shows source in global, arrow with dma.copy label, destination in local/shared.
Highlights the future handle returned.
"""
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from theme import parse_theme

from manim import *

C, THEME = parse_theme()


class DmaCopy(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text("dma.copy — Bulk Memory Transfer", font_size=26,
                      color=C["fg"], font="Monospace")
        title.to_edge(UP, buff=0.3)
        self.add(title)

        # Source: global memory
        src_box = Rectangle(width=4.5, height=1.6, fill_color=C["global_c"],
                            fill_opacity=0.12, stroke_color=C["global_c"], stroke_width=2)
        src_box.move_to(LEFT * 3 + UP * 1.0)
        src_lbl = Text("Global Memory", font_size=16, color=C["global_c"], font="Monospace")
        src_lbl.move_to(src_box.get_top() + DOWN * 0.25)

        src_cells = VGroup()
        for i in range(6):
            r = Rectangle(width=0.55, height=0.45, fill_color=C["global_c"],
                          fill_opacity=0.4, stroke_color=C["global_c"], stroke_width=1)
            r.move_to(LEFT * 3 + LEFT * 1.5 + RIGHT * i * 0.6 + UP * 0.85)
            v = Text(f"a{i}", font_size=12, color=C["fg"], font="Monospace").move_to(r)
            src_cells.add(VGroup(r, v))
        dots_s = Text("...", font_size=14, color=C["fg2"], font="Monospace")
        dots_s.next_to(src_cells, RIGHT, buff=0.08)

        src_code = Text("lhs.chunkat(tile)", font_size=12, color=C["global_c"], font="Monospace")
        src_code.move_to(src_box.get_bottom() + UP * 0.15)

        self.add(src_box, src_lbl, src_cells, dots_s, src_code)

        # Destination: local memory
        dst_box = Rectangle(width=4.5, height=1.6, fill_color=C["local_c"],
                            fill_opacity=0.12, stroke_color=C["local_c"], stroke_width=2)
        dst_box.move_to(RIGHT * 3.5 + UP * 1.0)
        dst_lbl = Text("Local Memory", font_size=16, color=C["local_c"], font="Monospace")
        dst_lbl.move_to(dst_box.get_top() + DOWN * 0.25)

        # DMA arrow — from global box right edge to local box left edge
        dma_arrow = Arrow(src_box.get_right(), dst_box.get_left(),
                          buff=0.1, stroke_width=4, color=C["arrow"],
                          max_tip_length_to_length_ratio=0.06)
        dma_lbl = Text("dma.copy", font_size=18, color=C["arrow"], font="Monospace")
        dma_lbl.next_to(dma_arrow, UP, buff=0.15)
        arrow_sub = Text("=> local", font_size=14, color=C["local_c"], font="Monospace")
        arrow_sub.next_to(dma_arrow, DOWN, buff=0.1)
        self.add(dma_arrow, dma_lbl, arrow_sub)

        dst_cells = VGroup()
        for i in range(6):
            r = Rectangle(width=0.55, height=0.45, fill_color=C["local_c"],
                          fill_opacity=0.4, stroke_color=C["local_c"], stroke_width=1)
            r.move_to(RIGHT * 3.5 + LEFT * 1.5 + RIGHT * i * 0.6 + UP * 0.85)
            v = Text(f"a{i}", font_size=12, color=C["fg"], font="Monospace").move_to(r)
            dst_cells.add(VGroup(r, v))
        dots_d = Text("...", font_size=14, color=C["fg2"], font="Monospace")
        dots_d.next_to(dst_cells, RIGHT, buff=0.08)

        self.add(dst_box, dst_lbl, dst_cells, dots_d)

        # Future handle
        future_box = Rectangle(width=5.5, height=1.6, fill_color=C["future_c"],
                                fill_opacity=0.1, stroke_color=C["future_c"], stroke_width=2)
        future_box.move_to(DOWN * 1.5)
        future_lbl = Text("DMA Future: lhs_load", font_size=16, color=C["future_c"],
                           font="Monospace")
        future_lbl.move_to(future_box.get_top() + DOWN * 0.3)

        future_fields = VGroup()
        for i, (field, desc) in enumerate([
            (".data", "access the copied tensor"),
            (".status", "check if transfer is complete"),
        ]):
            f = Text(f"{field}  →  {desc}", font_size=12, color=C["fg2"], font="Monospace")
            f.move_to(future_box.get_center() + DOWN * (i - 0.3) * 0.35)
            future_fields.add(f)

        result_arrow = Arrow([0, src_box.get_bottom()[1], 0], future_box.get_top(),
                              buff=0.5, stroke_width=2, color=C["future_c"],
                              max_tip_length_to_length_ratio=0.1)
        result_lbl = Text("returns", font_size=13, color=C["future_c"], font="Monospace")
        result_lbl.next_to(result_arrow, RIGHT, buff=0.1)

        self.add(future_box, future_lbl, future_fields, result_arrow, result_lbl)

        # Code at bottom
        code = Text(
            "lhs_load = dma.copy lhs.chunkat(tile) => local;",
            font_size=14, color=C["fg2"], font="Monospace"
        )
        code.move_to(DOWN * 3.0)
        self.add(code)
