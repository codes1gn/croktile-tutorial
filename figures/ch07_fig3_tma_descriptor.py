"""
Figure 3 (ch07): TMA descriptor structure.
Shows how a TMA descriptor encodes base pointer, dimensions, strides,
and how the hardware unit consumes it to move a tile into SMEM.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from theme import parse_theme
from manim import *

C, THEME = parse_theme()


class TMADescriptor(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text("TMA Descriptor -> Hardware Tile Fetch", font_size=20, color=C["fg"], font="Monospace")
        title.to_edge(UP, buff=0.3)
        self.add(title)

        # Descriptor fields (left)
        desc_x = -4.0
        desc_title = Text("Tensor Descriptor", font_size=13, color=C["fg2"], font="Monospace")
        desc_title.move_to(RIGHT * desc_x + UP * 1.8)
        self.add(desc_title)

        fields = [
            ("base_ptr", C["orange"]),
            ("dim[0] = M", C["blue"]),
            ("dim[1] = K", C["blue"]),
            ("stride[0]", C["green"]),
            ("stride[1]", C["green"]),
            ("swizzle", C["purple"]),
        ]
        for i, (name, col) in enumerate(fields):
            y = 1.2 - i * 0.42
            r = Rectangle(width=2.6, height=0.36, fill_color=col, fill_opacity=0.2,
                          stroke_color=col, stroke_width=1.2)
            r.move_to(RIGHT * desc_x + UP * y)
            self.add(r)
            t = Text(name, font_size=13, color=C["fg"], font="Monospace")
            t.move_to(r)
            self.add(t)

        # Arrow from descriptor to TMA unit
        arrow1 = Arrow(RIGHT * (desc_x + 1.5) + DOWN * 0.4,
                       RIGHT * -0.8 + DOWN * 0.4,
                       color=C["arrow"], stroke_width=2, buff=0.1, max_tip_length_to_length_ratio=0.15)
        self.add(arrow1)

        # TMA unit (center)
        tma_box = RoundedRectangle(width=2.4, height=1.6, corner_radius=0.15,
                                    fill_color=C["fill"], fill_opacity=0.6,
                                    stroke_color=C["fg3"], stroke_width=1.5)
        tma_box.move_to(RIGHT * 0.5 + DOWN * 0.4)
        self.add(tma_box)

        tma_label = Text("TMA Unit", font_size=14, color=C["fg2"], font="Monospace")
        tma_label.move_to(tma_box.get_center() + UP * 0.35)
        self.add(tma_label)

        tma_sub = Text("(near L2/SMEM)", font_size=12, color=C["dim"], font="Monospace")
        tma_sub.move_to(tma_box.get_center() + DOWN * 0.1)
        self.add(tma_sub)

        tma_detail = Text("HW multi-dim addr", font_size=12, color=C["fg3"], font="Monospace")
        tma_detail.move_to(tma_box.get_center() + DOWN * 0.45)
        self.add(tma_detail)

        # Arrow from TMA to SMEM tile (right)
        arrow2 = Arrow(RIGHT * 1.9 + DOWN * 0.4,
                       RIGHT * 3.5 + DOWN * 0.4,
                       color=C["arrow"], stroke_width=2, buff=0.1, max_tip_length_to_length_ratio=0.15)
        self.add(arrow2)

        tile_label = Text("one tile instruction", font_size=12, color=C["dim"], font="Monospace")
        tile_label.next_to(arrow2, UP, buff=0.05)
        self.add(tile_label)

        # SMEM tile (right)
        smem_title = Text("Shared Memory", font_size=13, color=C["fg2"], font="Monospace")
        smem_title.move_to(RIGHT * 4.5 + UP * 1.0)
        self.add(smem_title)

        tile_grid = VGroup()
        for r in range(4):
            for c in range(4):
                sq = Square(side_length=0.38, fill_color=C["smem_c"], fill_opacity=0.3,
                            stroke_color=C["smem_c"], stroke_width=1)
                sq.move_to(RIGHT * (3.6 + c * 0.42) + UP * (0.5 - r * 0.42))
                tile_grid.add(sq)
        self.add(tile_grid)

        tile_dim = Text("[WARP_M, TILE_K]", font_size=12, color=C["dim"], font="Monospace")
        tile_dim.move_to(RIGHT * 4.5 + DOWN * 1.0)
        self.add(tile_dim)

        # Global memory (top-right)
        gmem = Text("Global (HBM)", font_size=14, color=C["global_c"], font="Monospace")
        gmem.move_to(RIGHT * 4.5 + UP * 1.8)
        self.add(gmem)

        gmem_arrow = Arrow(RIGHT * 4.5 + UP * 1.55, RIGHT * 4.5 + UP * 1.15,
                           color=C["dim"], stroke_width=1.5, buff=0.05, max_tip_length_to_length_ratio=0.2)
        self.add(gmem_arrow)

        # Bottom note
        note = Text(
            "DMA: threads cooperate on address math     TMA: one descriptor, hardware does the rest",
            font_size=13, color=C["fg3"], font="Monospace",
        )
        note.to_edge(DOWN, buff=0.3)
        self.add(note)
