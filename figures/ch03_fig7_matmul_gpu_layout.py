"""
Figure 7: Matmul GPU Layout — how the nested parallel maps to actual GPU.
Shows SMs with blocks assigned, threads within a block, and shared memory.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from theme import parse_theme
from manim import *

C, THEME = parse_theme()


class MatmulGpuLayout(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text("Matmul: Code → GPU Resource Layout", font_size=20,
                      color=C["fg"], font="Monospace")
        title.to_edge(UP, buff=0.25)
        self.add(title)

        # --- Left: Code structure ---
        code_title = Text("Croqtile Code", font_size=14, color=C["blue"],
                           font="Monospace")
        code_title.move_to(LEFT * 4.5 + UP * 2.2)
        self.add(code_title)

        code_lines = [
            ("parallel {px,py} by [8,16] : block", C["orange"], 0),
            ("  foreach {tile_k} in [16]", C["purple"], 1),
            ("    dma.copy ... => shared", C["arrow"], 2),
            ("    parallel {qx,qy} by [16,16] : thread", C["green"], 2),
            ("      foreach k in [16]", C["teal"], 3),
            ("        output += lhs × rhs", C["fg2"], 4),
        ]

        for i, (text, color, indent) in enumerate(code_lines):
            t = Text(text, font_size=12, color=color, font="Monospace")
            t.move_to(LEFT * (4.5 - indent * 0.2) + UP * (1.5 - i * 0.45))
            t.align_to(LEFT * (6.5 - indent * 0.3), LEFT)
            self.add(t)

        # --- Right: GPU hardware ---
        gpu_title = Text("GPU Hardware", font_size=14, color=C["green"],
                          font="Monospace")
        gpu_title.move_to(RIGHT * 3.0 + UP * 2.2)
        self.add(gpu_title)

        # GPU chip outline
        gpu_box = Rectangle(width=7.0, height=4.5, fill_color=C["fill"],
                             fill_opacity=0.5, stroke_color=C["fg3"], stroke_width=1)
        gpu_box.move_to(RIGHT * 3.0 + DOWN * 0.5)
        self.add(gpu_box)

        # SMs with blocks
        sm_data = [
            (RIGHT * 1.0 + UP * 0.8, "SM 0", ["(0,0)", "(0,1)"]),
            (RIGHT * 3.5 + UP * 0.8, "SM 1", ["(0,2)", "(0,3)"]),
            (RIGHT * 1.0 + DOWN * 1.5, "SM 2", ["(1,0)", "(1,1)"]),
            (RIGHT * 3.5 + DOWN * 1.5, "SM 3", ["(1,2)", "(1,3)"]),
        ]
        block_target = None
        thread_target = None

        for pos, sm_name, blocks in sm_data:
            sm = Rectangle(width=2.6, height=2.0, fill_color=C["sm_c"],
                           fill_opacity=0.1, stroke_color=C["sm_c"], stroke_width=1)
            sm.move_to(pos)
            sm_lbl = Text(sm_name, font_size=12, color=C["sm_c"],
                          font="Monospace")
            sm_lbl.move_to(sm.get_top() + DOWN * 0.15)
            self.add(sm, sm_lbl)

            # Shared memory bar
            smem = Rectangle(width=2.2, height=0.3, fill_color=C["shared_c"],
                              fill_opacity=0.25, stroke_color=C["shared_c"],
                              stroke_width=1)
            smem.move_to(pos + DOWN * 0.0)
            smem_l = Text("SMEM", font_size=12, color=C["shared_c"],
                           font="Monospace").move_to(smem)
            self.add(smem, smem_l)

            # Block labels — stacked vertically to avoid overlap
            for j, blk in enumerate(blocks):
                bt = Text(f"block{blk}", font_size=10, color=C["orange"],
                           font="Monospace")
                bt.move_to(pos + LEFT * 0.55 + RIGHT * j * 1.1 + DOWN * 0.45)
                self.add(bt)

            # Thread dots
            for tx in range(4):
                for ty in range(2):
                    dot = Dot(point=pos + LEFT * 0.6 + RIGHT * tx * 0.35 +
                              DOWN * 0.7 + DOWN * ty * 0.2,
                              radius=0.04, color=C["green"])
                    self.add(dot)

            # Use SM 0 as anchor targets for mapping arrows
            if sm_name == "SM 0":
                block_target = pos + LEFT * 1.0 + DOWN * 0.45
                thread_target = pos + LEFT * 0.7 + DOWN * 0.8

        # Arrows from code to hardware
        arr1 = Arrow(LEFT * 3.1 + UP * 1.55, block_target,
                     buff=0.1, stroke_width=2, color=C["orange"],
                     max_tip_length_to_length_ratio=0.06)
        a1_lbl = Text(": block", font_size=12, color=C["orange"],
                       font="Monospace")
        a1_lbl.next_to(arr1, UP, buff=0.03)
        self.add(arr1, a1_lbl)

        arr2 = Arrow(LEFT * 2.3 + UP * 0.17, thread_target,
                     buff=0.1, stroke_width=2, color=C["green"],
                     max_tip_length_to_length_ratio=0.06)
        a2_lbl = Text(": thread", font_size=12, color=C["green"],
                       font="Monospace")
        a2_lbl.next_to(arr2, DOWN, buff=0.03)
        self.add(arr2, a2_lbl)

        # Note
        note = Text("128 blocks total (8×16), each with 256 threads (16×16)\n"
                     "showing 4 SMs with 2 blocks each",
                     font_size=12, color=C["dim"], font="Monospace")
        note.to_edge(DOWN, buff=0.3)
        self.add(note)
