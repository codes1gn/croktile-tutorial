"""
Ch07 Fig8: Expressiveness vs Performance sweet-spot diagram.
CUDA's wide expressible range includes many slow patterns;
Croqtile's restricted range maps entirely to the performance sweet spot.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from theme import parse_theme
from manim import *

C, THEME = parse_theme()


class ExpressivenessVsPerformance(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text(
            "Expressiveness vs Performance: data movement patterns",
            font_size=20, color=C["fg"], font="Monospace",
        )
        title.to_edge(UP, buff=0.3)
        self.add(title)

        # Large outer ellipse: CUDA's full expressible range
        cuda_ellipse = Ellipse(
            width=9.0, height=4.5,
            color=C["orange"], stroke_width=2.5,
            fill_color=C["orange"], fill_opacity=0.06,
        )
        cuda_ellipse.move_to(DOWN * 0.3)
        self.add(cuda_ellipse)

        cuda_lbl = Text(
            "CUDA expressible range",
            font_size=16, color=C["orange"], font="Monospace",
        )
        cuda_lbl.move_to(cuda_ellipse.get_top() + DOWN * 0.35)
        self.add(cuda_lbl)

        # Inner filled region: performance sweet spot
        sweet_ellipse = Ellipse(
            width=4.2, height=2.8,
            color=C["green"], stroke_width=2.5,
            fill_color=C["green"], fill_opacity=0.15,
        )
        sweet_ellipse.move_to(DOWN * 0.3 + LEFT * 0.3)
        self.add(sweet_ellipse)

        sweet_lbl = Text(
            "Performance sweet spot",
            font_size=15, color=C["green"], font="Monospace",
        )
        sweet_lbl.move_to(sweet_ellipse.get_center() + UP * 0.9)
        self.add(sweet_lbl)

        sweet_sub = Text(
            "coalesced loads · conflict-free SMEM · aligned swizzle",
            font_size=10, color=C["fg3"], font="Monospace",
        )
        sweet_sub.move_to(sweet_ellipse.get_center() + UP * 0.55)
        self.add(sweet_sub)

        # Croqtile range: entirely inside sweet spot
        croq_rect = RoundedRectangle(
            width=3.0, height=1.6, corner_radius=0.15,
            color=C["blue"], stroke_width=3,
            fill_color=C["blue"], fill_opacity=0.12,
        )
        croq_rect.move_to(sweet_ellipse.get_center() + DOWN * 0.15)
        self.add(croq_rect)

        croq_lbl = Text(
            "Croqtile range",
            font_size=15, color=C["blue"], font="Monospace",
        )
        croq_lbl.move_to(croq_rect.get_center() + UP * 0.25)
        self.add(croq_lbl)

        croq_sub = Text(
            "dma.copy · tma.copy · swiz<N>",
            font_size=11, color=C["fg2"], font="Monospace",
        )
        croq_sub.move_to(croq_rect.get_center() + DOWN * 0.1)
        self.add(croq_sub)

        croq_note = Text(
            "always fast — by construction",
            font_size=10, color=C["blue"], font="Monospace",
        )
        croq_note.move_to(croq_rect.get_center() + DOWN * 0.45)
        self.add(croq_note)

        # Bad patterns scattered in the outer region (outside sweet spot)
        bad_patterns = [
            ("strided uncoalesced", RIGHT * 3.2 + DOWN * 0.9),
            ("bank conflicts", RIGHT * 3.0 + UP * 0.3),
            ("misaligned swizzle", RIGHT * 2.5 + DOWN * 1.6),
            ("scalar loads", LEFT * 3.5 + DOWN * 1.5),
            ("warp-divergent addr", LEFT * 3.0 + UP * 0.5),
        ]
        for txt, pos in bad_patterns:
            x_mark = Text("✗", font_size=14, color=C["red"], font="Monospace")
            x_mark.move_to(pos + DOWN * 0.3)
            label = Text(txt, font_size=9, color=C["red"], font="Monospace")
            label.next_to(x_mark, DOWN, buff=0.06)
            self.add(x_mark, label)

        # TMA subset label
        tma_note = Text(
            "TMA: fixed descriptor patterns",
            font_size=10, color=C["purple"], font="Monospace",
        )
        tma_note.move_to(sweet_ellipse.get_center() + DOWN * 1.15 + LEFT * 0.3)
        self.add(tma_note)

        # Footer
        foot = Text(
            "Croqtile trades expressiveness for guaranteed performance:"
            " every pattern it can express is in the sweet spot",
            font_size=10, color=C["dim"], font="Monospace",
        )
        foot.to_edge(DOWN, buff=0.25)
        self.add(foot)
