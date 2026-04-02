"""
Figure: GPU Memory Hierarchy — Croktile specifiers mapped to GPU hardware.
Shows the physical GPU layout (DRAM, L2, SM with SMEM and registers)
and how Croktile's global/shared/local map to each level.
"""
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from theme import parse_theme

from manim import *

C, THEME = parse_theme()


class MemoryHierarchy(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text("Memory Specifiers → GPU Hardware", font_size=26,
                      color=C["fg"], font="Monospace")
        title.to_edge(UP, buff=0.3)
        self.add(title)

        # GPU side (left)
        gpu_label = Text("GPU Hardware", font_size=18, color=C["label_c"], font="Monospace")
        gpu_label.move_to(LEFT * 3.2 + UP * 2.3)
        self.add(gpu_label)

        # DRAM
        dram = Rectangle(width=5.5, height=1.0, fill_color=C["dram_c"],
                          fill_opacity=0.5, stroke_color=C["global_c"], stroke_width=2)
        dram.move_to(LEFT * 3.2 + UP * 1.4)
        dram_lbl = Text("HBM / DRAM  (Global Memory)", font_size=13,
                         color=C["fg"], font="Monospace").move_to(dram)
        dram_size = Text("~80 GB, ~2 TB/s", font_size=10,
                          color=C["label_c"], font="Monospace")
        dram_size.next_to(dram, DOWN, buff=0.05)
        self.add(dram, dram_lbl, dram_size)

        # L2 Cache
        l2 = Rectangle(width=4.5, height=0.6, fill_color=C["l2_c"],
                        fill_opacity=0.4, stroke_color=C["fg3"], stroke_width=1)
        l2.move_to(LEFT * 3.2 + UP * 0.2)
        l2_lbl = Text("L2 Cache (hardware-managed)", font_size=11,
                       color=C["label_c"], font="Monospace").move_to(l2)
        self.add(l2, l2_lbl)

        # SMs
        sm_y = -1.2
        for sm_idx in range(2):
            sm_x = LEFT * 3.2 + RIGHT * (sm_idx - 0.5) * 2.8

            sm_box = Rectangle(width=2.5, height=2.5, fill_color=C["sm_c"],
                                fill_opacity=0.15, stroke_color=C["sm_c"], stroke_width=1.5)
            sm_box.move_to(sm_x + DOWN * 1.2)

            sm_title = Text(f"SM {sm_idx}", font_size=12, color=C["sm_c"], font="Monospace")
            sm_title.move_to(sm_box.get_top() + DOWN * 0.2)

            smem = Rectangle(width=2.1, height=0.6, fill_color=C["smem_c"],
                              fill_opacity=0.4, stroke_color=C["shared_c"], stroke_width=1.5)
            smem.move_to(sm_x + DOWN * 0.7)
            smem_lbl = Text("Shared Memory (SMEM)", font_size=9,
                             color=C["fg"], font="Monospace").move_to(smem)
            smem_size = Text("~228 KB", font_size=8, color=C["label_c"],
                              font="Monospace").next_to(smem, DOWN, buff=0.03)

            regs = VGroup()
            for t in range(4):
                r = Rectangle(width=0.4, height=0.4, fill_color=C["reg_c"],
                                fill_opacity=0.4, stroke_color=C["local_c"], stroke_width=1)
                r.move_to(sm_x + RIGHT * (t - 1.5) * 0.5 + DOWN * 1.7)
                rl = Text(f"R{t}", font_size=7, color=C["fg"], font="Monospace").move_to(r)
                regs.add(VGroup(r, rl))

            reg_label = Text("Registers (per-thread)", font_size=9,
                              color=C["label_c"], font="Monospace")
            reg_label.move_to(sm_x + DOWN * 2.2)

            self.add(sm_box, sm_title, smem, smem_lbl, smem_size, regs, reg_label)

        # Croktile side (right)
        crk_label = Text("Croktile Specifiers", font_size=18, color=C["label_c"], font="Monospace")
        crk_label.move_to(RIGHT * 3.5 + UP * 2.3)
        self.add(crk_label)

        specs = [
            ("global", C["global_c"], UP * 1.4, "=> global", "Full device memory\nAll threads, all blocks"),
            ("shared", C["shared_c"], DOWN * 0.7, "=> shared", "Block-scoped SRAM\nAll threads in one block"),
            ("local", C["local_c"], DOWN * 1.7, "=> local", "Thread-private\nRegisters / local scratch"),
        ]

        for name, color, yoff, syntax, desc in specs:
            box = Rectangle(width=3.5, height=1.0, fill_color=color,
                            fill_opacity=0.15, stroke_color=color, stroke_width=2)
            box.move_to(RIGHT * 3.5 + yoff)

            name_t = Text(syntax, font_size=16, color=color, font="Monospace")
            name_t.move_to(box.get_top() + DOWN * 0.25)

            desc_t = Text(desc, font_size=10, color=C["label_c"], font="Monospace",
                           line_spacing=1.2)
            desc_t.move_to(box.get_bottom() + UP * 0.25)

            self.add(box, name_t, desc_t)

        # Arrows connecting Croktile specifiers to GPU hardware
        arrows_data = [
            (RIGHT * 1.7 + UP * 1.4, dram.get_right(), C["global_c"]),
            (RIGHT * 1.7 + DOWN * 0.7, LEFT * 3.2 + DOWN * 0.7, C["shared_c"]),
            (RIGHT * 1.7 + DOWN * 1.7, LEFT * 3.2 + DOWN * 1.7, C["local_c"]),
        ]
        for src, dst, color in arrows_data:
            arr = Arrow(src, dst, buff=0.15, stroke_width=2, color=color,
                        max_tip_length_to_length_ratio=0.06)
            self.add(arr)

        # Bandwidth annotation
        bw = Text("← faster, smaller →\n← slower, larger →",
                   font_size=10, color=C["fg3"], font="Monospace", line_spacing=1.2)
        bw.move_to(LEFT * 3.2 + DOWN * 3.2)
        self.add(bw)
