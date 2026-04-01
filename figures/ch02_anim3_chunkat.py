"""
Animation 3: chunkat Semantics — animated 2D tile sweep.
Shows a 2D grid, sweeps across tiles highlighting each in sequence,
then zooms into one to show the sub-tensor and index composition.
"""
from manim import *

BG = "#1a1a2e"
GRID_C = "#37474F"
TILE_C = "#4CAF50"
HL_C = "#FFEB3B"
DIM_C = "#78909C"


class ChunkatAnim(Scene):
    def construct(self):
        self.camera.background_color = BG

        title = Text("chunkat — 2D Tile Selection", font_size=28,
                      color=WHITE, font="Monospace")
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.5)

        tensor_lbl = Text("tensor [64, 128]", font_size=16, color=GREY_B, font="Monospace")
        tensor_lbl.move_to(UP * 2.5)
        self.play(FadeIn(tensor_lbl), run_time=0.3)

        rows, cols = 4, 8
        cw, ch = 0.7, 0.6
        ox = -cols * cw / 2 + cw / 2
        oy = 1.2

        grid = VGroup()
        cell_map = {}
        for r in range(rows):
            for c in range(cols):
                rect = Rectangle(width=cw - 0.04, height=ch - 0.04,
                                 fill_color=GRID_C, fill_opacity=0.3,
                                 stroke_color=GREY_D, stroke_width=0.8)
                x = ox + c * cw
                y = oy - r * ch
                rect.move_to([x, y, 0])
                lbl = Text(f"({r},{c})", font_size=9, color=DIM_C,
                           font="Monospace").move_to(rect)
                cell = VGroup(rect, lbl)
                grid.add(cell)
                cell_map[(r, c)] = cell

        # Row/col labels
        axis_labels = VGroup()
        for r in range(rows):
            t = Text(f"tr={r}", font_size=11, color=DIM_C, font="Monospace")
            t.move_to([ox - cw * 0.8, oy - r * ch, 0])
            axis_labels.add(t)
        for c in range(cols):
            t = Text(f"{c}", font_size=10, color=DIM_C, font="Monospace")
            t.move_to([ox + c * cw, oy + rows * ch / 2 + 0.05, 0])
            axis_labels.add(t)

        self.play(*[FadeIn(c) for c in grid], *[FadeIn(a) for a in axis_labels],
                  run_time=0.6)
        self.wait(0.3)

        # Sweep: highlight tiles in sequence
        sweep_text = Text("parallel {tr, tc} by [4, 8] — sweep all tiles",
                          font_size=14, color=GREY_B, font="Monospace")
        sweep_text.move_to(DOWN * 1.5)
        self.play(FadeIn(sweep_text), run_time=0.3)

        prev_cell = None
        for r in range(rows):
            for c in range(cols):
                cell = cell_map[(r, c)]
                anims = [cell[0].animate.set_fill(HL_C, 0.6).set_stroke(HL_C, 2)]
                if prev_cell:
                    anims.append(prev_cell[0].animate.set_fill(TILE_C, 0.25).set_stroke(TILE_C, 1))
                self.play(*anims, run_time=0.06)
                prev_cell = cell

        if prev_cell:
            self.play(prev_cell[0].animate.set_fill(TILE_C, 0.25).set_stroke(TILE_C, 1),
                      run_time=0.1)

        self.wait(0.5)

        # Focus on (1,3)
        focus_text = Text("Focus: chunkat(1, 3)", font_size=18,
                          color=HL_C, font="Monospace")
        self.play(
            FadeOut(sweep_text),
            FadeIn(focus_text.move_to(DOWN * 1.5)),
            cell_map[(1, 3)][0].animate.set_fill(HL_C, 0.8).set_stroke(HL_C, 3),
            run_time=0.5
        )
        self.wait(0.5)

        # Zoom panel
        zoom_box = Rectangle(width=2.8, height=2.0,
                             fill_color=HL_C, fill_opacity=0.1,
                             stroke_color=HL_C, stroke_width=2)
        zoom_box.move_to(RIGHT * 4 + DOWN * 1.0)

        zoom_title = Text("[16, 16] sub-tensor", font_size=14,
                          color=TILE_C, font="Monospace")
        zoom_title.next_to(zoom_box, UP, buff=0.15)

        mini_grid = VGroup()
        mini_cw, mini_ch = 0.35, 0.28
        mini_ox = RIGHT * 4 + LEFT * 1.0 + UP * 0.2
        for mr in range(4):
            for mc in range(4):
                r = Rectangle(width=mini_cw - 0.02, height=mini_ch - 0.02,
                              fill_color=TILE_C, fill_opacity=0.4,
                              stroke_color=TILE_C, stroke_width=0.8)
                pos = mini_ox + RIGHT * mc * mini_cw + DOWN * mr * mini_ch
                r.move_to(pos)
                real_r = 16 + mr
                real_c = 48 + mc
                v = Text(f"{real_r},{real_c}", font_size=5, color=WHITE,
                         font="Monospace").move_to(r)
                mini_grid.add(VGroup(r, v))

        dots = Text("... (16x16)", font_size=11, color=GREY_B, font="Monospace")
        dots.move_to(RIGHT * 4 + DOWN * 1.5)

        arrow = Arrow(cell_map[(1, 3)][0].get_right(),
                      zoom_box.get_left(),
                      buff=0.15, stroke_width=2, color=HL_C,
                      max_tip_length_to_length_ratio=0.06)

        self.play(
            GrowArrow(arrow),
            FadeIn(zoom_box), FadeIn(zoom_title),
            *[FadeIn(c) for c in mini_grid],
            FadeIn(dots),
            run_time=0.8
        )

        # Index formula
        formula = Text(
            "output.at(tr # i, tc # j)\n= output.at(1*16+i, 3*16+j)",
            font_size=12, color=DIM_C, font="Monospace"
        )
        formula.move_to(RIGHT * 4 + DOWN * 2.6)
        self.play(FadeIn(formula), run_time=0.4)
        self.wait(2.5)
