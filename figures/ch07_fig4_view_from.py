"""
Figure 4 (ch07): view(M,N).from(r,c) vs chunkat.
Left: chunkat selects aligned grid cells.
Right: view/from selects an arbitrary-offset window.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from theme import parse_theme
from manim import *

C, THEME = parse_theme()


class ViewFrom(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text("chunkat (aligned) vs view/from (arbitrary offset)", font_size=18, color=C["fg"], font="Monospace")
        title.to_edge(UP, buff=0.3)
        self.add(title)

        cell = 0.32

        def draw_grid(origin, rows, cols, label, highlight=None, hl_color=None, hl_label=None):
            grp = VGroup()
            for r in range(rows):
                for c in range(cols):
                    sq = Square(side_length=cell, fill_color=C["fill"], fill_opacity=0.3,
                                stroke_color=C["stroke"], stroke_width=0.8)
                    sq.move_to(origin + RIGHT * c * cell + DOWN * r * cell)
                    grp.add(sq)
            self.add(grp)

            lbl = Text(label, font_size=12, color=C["fg2"], font="Monospace")
            lbl.next_to(grp, UP, buff=0.15)
            self.add(lbl)

            if highlight and hl_color:
                r0, c0, rh, ch = highlight
                hl_rect = Rectangle(width=ch * cell, height=rh * cell,
                                     fill_color=hl_color, fill_opacity=0.35,
                                     stroke_color=hl_color, stroke_width=2)
                hl_rect.move_to(origin + RIGHT * (c0 * cell + (ch - 1) * cell / 2) +
                                DOWN * (r0 * cell + (rh - 1) * cell / 2))
                self.add(hl_rect)
                if hl_label:
                    hl_t = Text(hl_label, font_size=12, color=hl_color, font="Monospace")
                    hl_t.next_to(hl_rect, DOWN, buff=0.08)
                    self.add(hl_t)

        # Left: chunkat - aligned tiles
        left_origin = LEFT * 4.5 + UP * 1.0
        draw_grid(left_origin, 8, 8, "chunkat(i, j)", highlight=(0, 0, 4, 4),
                  hl_color=C["blue"], hl_label=None)
        draw_grid(left_origin, 8, 8, "", highlight=(0, 4, 4, 4),
                  hl_color=C["green"], hl_label=None)
        draw_grid(left_origin, 8, 8, "", highlight=(4, 0, 4, 4),
                  hl_color=C["orange"], hl_label=None)
        draw_grid(left_origin, 8, 8, "", highlight=(4, 4, 4, 4),
                  hl_color=C["purple"], hl_label=None)
        
        ca00_note = Text("chunk (0,0)", font_size=12, color=C["blue"], font="Monospace")
        ca00_note.move_to(left_origin + RIGHT * 0.5 + DOWN * 0.5)
        self.add(ca00_note)

        left_note = Text("tiles must align to grid", font_size=13, color=C["dim"], font="Monospace")
        left_note.move_to(left_origin + DOWN * 2.7 + RIGHT * 1.1)
        self.add(left_note)

        # Right: view/from - arbitrary offset
        right_origin = RIGHT * 1.5 + UP * 1.0
        draw_grid(right_origin, 8, 8, "view(4,4).from(2,3)", highlight=(2, 3, 4, 4),
                  hl_color=C["red"], hl_label="window at (2,3)")

        right_note = Text("origin is arbitrary (row=2, col=3)", font_size=13, color=C["dim"], font="Monospace")
        right_note.move_to(right_origin + DOWN * 2.7 + RIGHT * 1.2)
        self.add(right_note)

        # Code comparison at bottom
        code_left = Text('lhs.chunkat(i, j)', font_size=12, color=C["blue"], font="Monospace")
        code_left.move_to(LEFT * 1.3 + DOWN * 2.7)
        self.add(code_left)

        vs = Text("vs", font_size=12, color=C["dim"], font="Monospace")
        vs.move_to(DOWN * 2.7)
        self.add(vs)

        code_right = Text('lhs.view(4,4).from(2,3)', font_size=12, color=C["red"], font="Monospace")
        code_right.move_to(RIGHT * 1.5 + DOWN * 2.7)
        self.add(code_right)
