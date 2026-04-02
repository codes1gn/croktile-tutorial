"""
Figure 2: GPU tensor core register layout — fragmented lane ownership.
Threads in a warp own scattered pieces of the tile; Croktile hides this.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from manim import *
from theme import parse_theme

C, THEME = parse_theme()


class RegisterLoading(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text(
            "Tensor Core Register Layout (simplified)",
            font_size=20, color=C["fg"], font="Monospace",
        )
        title.to_edge(UP, buff=0.3)
        self.add(title)

        sub = Text(
            "Each thread in a warp owns scattered register fragments of the tile",
            font_size=12, color=C["fg3"], font="Monospace",
        )
        sub.next_to(title, DOWN, buff=0.15)
        self.add(sub)

        cell = 0.38
        rows, cols = 8, 8
        colors_per_thread = [
            C["blue"], C["orange"], C["green"], C["purple"],
            C["red"], C["teal"], C["yellow"], C["pink"],
        ]
        ownership = [
            [0,0,1,1,2,2,3,3],
            [0,0,1,1,2,2,3,3],
            [4,4,5,5,6,6,7,7],
            [4,4,5,5,6,6,7,7],
            [0,0,1,1,2,2,3,3],
            [0,0,1,1,2,2,3,3],
            [4,4,5,5,6,6,7,7],
            [4,4,5,5,6,6,7,7],
        ]

        grid_center = LEFT * 2.0 + DOWN * 0.6
        grid = VGroup()
        for r in range(rows):
            for c_idx in range(cols):
                sq = Square(side_length=cell,
                            fill_color=colors_per_thread[ownership[r][c_idx]],
                            fill_opacity=0.3,
                            stroke_color=colors_per_thread[ownership[r][c_idx]],
                            stroke_width=1.2)
                sq.move_to(grid_center + RIGHT * (c_idx - cols/2 + 0.5) * cell
                           + DOWN * (r - rows/2 + 0.5) * cell)
                grid.add(sq)
        self.add(grid)

        tile_label = Text("8×8 MMA tile", font_size=12, color=C["fg2"], font="Monospace")
        tile_label.next_to(grid, DOWN, buff=0.15)
        self.add(tile_label)

        legend_title = Text("Thread ownership:", font_size=12, color=C["fg"], font="Monospace")
        legend_title.move_to(RIGHT * 2.8 + UP * 0.3)
        self.add(legend_title)
        for i in range(8):
            sq = Square(side_length=0.2,
                        fill_color=colors_per_thread[i], fill_opacity=0.4,
                        stroke_color=colors_per_thread[i], stroke_width=1)
            label = Text(f"T{i*4}–T{i*4+3}", font_size=9, color=C["fg2"], font="Monospace")
            row = VGroup(sq, label).arrange(RIGHT, buff=0.1)
            row.move_to(RIGHT * 2.8 + DOWN * (0.0 + i * 0.28))
            self.add(row)

        brace_l = Text("Croktile:", font_size=14, color=C["green"], font="Monospace")
        brace_r = Text("mma.load / mma.store hide this", font_size=12, color=C["fg2"], font="Monospace")
        note = VGroup(brace_l, brace_r).arrange(RIGHT, buff=0.15)
        note.to_edge(DOWN, buff=0.3)
        self.add(note)
