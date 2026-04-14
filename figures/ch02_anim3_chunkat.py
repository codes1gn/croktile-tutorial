"""
Animation 3: chunkat Semantics — animated 2D tile sweep.
Shows a 2D grid, sweeps across tiles highlighting each in sequence,
then zooms into one to show the sub-tensor and index composition.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from theme import parse_theme
from manim import *

C, THEME = parse_theme()


class ChunkatAnim(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text("chunkat — 2D Tile Selection", font_size=28,
                      color=C["fg"], font="Monospace")
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.5)

        tensor_lbl = Text("tensor [64, 128]", font_size=16, color=C["fg2"], font="Monospace")
        tensor_lbl.move_to(LEFT * 0.8 + UP * 2.8)
        self.play(FadeIn(tensor_lbl), run_time=0.3)

        rows, cols = 4, 8
        cw, ch = 0.7, 0.6
        grid_shift = -0.8
        ox = -cols * cw / 2 + cw / 2 + grid_shift
        oy = 1.8

        grid = VGroup()
        cell_map = {}
        for r in range(rows):
            for c in range(cols):
                rect = Rectangle(width=cw - 0.04, height=ch - 0.04,
                                 fill_color=C["grid_c"], fill_opacity=0.3,
                                 stroke_color=C["dim"], stroke_width=0.8)
                x = ox + c * cw
                y = oy - r * ch
                rect.move_to([x, y, 0])
                lbl = Text(f"({r},{c})", font_size=12, color=C["dim_c"],
                           font="Monospace").move_to(rect)
                cell = VGroup(rect, lbl)
                grid.add(cell)
                cell_map[(r, c)] = cell

        # Row/col labels
        axis_labels = VGroup()
        for r in range(rows):
            t = Text(f"tr={r}", font_size=12, color=C["dim_c"], font="Monospace")
            t.move_to([ox - cw * 0.8, oy - r * ch, 0])
            axis_labels.add(t)
        for c in range(cols):
            t = Text(f"{c}", font_size=12, color=C["dim_c"], font="Monospace")
            t.move_to([ox + c * cw, 2.4, 0])
            axis_labels.add(t)

        self.play(*[FadeIn(c) for c in grid], *[FadeIn(a) for a in axis_labels],
                  run_time=0.6)
        self.wait(0.3)

        # Sweep: highlight tiles in sequence
        sweep_text = Text("parallel {tr, tc} by [4, 8] — sweep all tiles",
                          font_size=14, color=C["fg2"], font="Monospace")
        sweep_text.move_to(DOWN * 1.5)
        self.play(FadeIn(sweep_text), run_time=0.3)

        prev_cell = None
        for r in range(rows):
            for c in range(cols):
                cell = cell_map[(r, c)]
                anims = [cell[0].animate.set_fill(C["hl_c"], 0.6).set_stroke(C["hl_c"], 2)]
                if prev_cell:
                    anims.append(prev_cell[0].animate.set_fill(C["tile"], 0.25).set_stroke(C["tile"], 1))
                self.play(*anims, run_time=0.06)
                prev_cell = cell

        if prev_cell:
            self.play(prev_cell[0].animate.set_fill(C["tile"], 0.25).set_stroke(C["tile"], 1),
                      run_time=0.1)

        self.wait(0.5)

        # Focus on (1,3)
        focus_text = Text("Focus: chunkat(1, 3)", font_size=18,
                          color=C["hl_c"], font="Monospace")
        self.play(
            FadeOut(sweep_text),
            FadeIn(focus_text.move_to(DOWN * 1.5)),
            cell_map[(1, 3)][0].animate.set_fill(C["hl_c"], 0.8).set_stroke(C["hl_c"], 3),
            run_time=0.5
        )
        self.wait(0.5)

        # Zoom panel
        mini_s = 0.5
        zx, zy = 4.5, -2.2
        zw = 4 * mini_s + 0.6
        zh = 4 * mini_s + 0.6
        zoom_box = Rectangle(width=zw, height=zh,
                             fill_color=C["hl_c"], fill_opacity=0.1,
                             stroke_color=C["hl_c"], stroke_width=2)
        zoom_box.move_to([zx, zy, 0])

        zoom_title = Text("[16, 16] sub-tensor", font_size=14,
                          color=C["tile"], font="Monospace")
        zoom_title.next_to(zoom_box, UP, buff=0.15)

        mini_grid = VGroup()
        mini_ox = zx - 3 * mini_s / 2
        mini_oy = zy + 3 * mini_s / 2
        for mr in range(4):
            for mc in range(4):
                r = Rectangle(width=mini_s - 0.02, height=mini_s - 0.02,
                              fill_color=C["tile"], fill_opacity=0.4,
                              stroke_color=C["tile"], stroke_width=0.8)
                gx = mini_ox + mc * mini_s
                gy = mini_oy - mr * mini_s
                r.move_to([gx, gy, 0])
                real_r = 16 + mr
                real_c = 48 + mc
                v = Text(f"({real_r},{real_c})", font_size=8, color=C["fg"],
                         font="Monospace").move_to(r)
                mini_grid.add(VGroup(r, v))

        dots = Text("[16, 16] sub-tensor", font_size=12, color=C["fg2"], font="Monospace")
        dots.next_to(zoom_box, DOWN, buff=0.1)

        arrow = Arrow(cell_map[(1, 3)][0].get_right(),
                      zoom_box.get_left(),
                      buff=0.15, stroke_width=2, color=C["hl_c"],
                      max_tip_length_to_length_ratio=0.06)

        self.play(
            GrowArrow(arrow),
            FadeIn(zoom_box), FadeIn(zoom_title),
            *[FadeIn(c) for c in mini_grid],
            FadeIn(dots),
            run_time=0.8
        )

        self.wait(2.5)
