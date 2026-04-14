"""
Figure: GPU Memory Hierarchy — Croqtile specifiers mapped to GPU hardware.
Shows the physical GPU layout (DRAM, L2, SM with SMEM and registers)
and how Croqtile's global/shared/local map to each level.
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

        gpu_cx = -2.8

        # GPU side header
        gpu_label = Text("GPU Hardware", font_size=18, color=C["label_c"], font="Monospace")
        gpu_label.move_to([gpu_cx, 2.5, 0])
        self.add(gpu_label)

        # DRAM (Global Memory)
        dram = Rectangle(width=5.5, height=0.8, fill_color=C["dram_c"],
                          fill_opacity=0.5, stroke_color=C["global_c"], stroke_width=2)
        dram.move_to([gpu_cx, 1.8, 0])
        dram_lbl = Text("HBM / DRAM  (Global Memory)", font_size=13,
                         color=C["fg"], font="Monospace").move_to(dram)
        dram_size = Text("~80 GB, ~2 TB/s", font_size=11,
                          color=C["label_c"], font="Monospace")
        dram_size.next_to(dram, DOWN, buff=0.04)
        self.add(dram, dram_lbl, dram_size)

        # L2 Cache
        l2 = Rectangle(width=4.5, height=0.5, fill_color=C["l2_c"],
                        fill_opacity=0.4, stroke_color=C["fg3"], stroke_width=1)
        l2.move_to([gpu_cx, 0.8, 0])
        l2_lbl = Text("L2 Cache (hardware-managed)", font_size=11,
                       color=C["label_c"], font="Monospace").move_to(l2)
        self.add(l2, l2_lbl)

        # SM boxes — smaller and lower
        sm_w, sm_h = 2.2, 2.2
        sm_gap = 0.3
        sm0_cx = gpu_cx - (sm_w + sm_gap) / 2
        sm1_cx = gpu_cx + (sm_w + sm_gap) / 2
        sm_top_y = 0.3

        shared_target = None
        local_target = None
        for sm_idx, sm_cx in enumerate([sm0_cx, sm1_cx]):
            sm_box = Rectangle(width=sm_w, height=sm_h, fill_color=C["sm_c"],
                                fill_opacity=0.15, stroke_color=C["sm_c"], stroke_width=1.5)
            sm_box.move_to([sm_cx, sm_top_y - sm_h / 2, 0])

            sm_title = Text(f"SM {sm_idx}", font_size=11, color=C["sm_c"], font="Monospace")
            sm_title.move_to([sm_cx, sm_top_y - 0.15, 0])

            smem = Rectangle(width=sm_w - 0.3, height=0.5, fill_color=C["smem_c"],
                              fill_opacity=0.4, stroke_color=C["shared_c"], stroke_width=1.5)
            smem.move_to([sm_cx, sm_top_y - 0.6, 0])
            smem_lbl = Text("Shared Memory", font_size=10,
                             color=C["fg"], font="Monospace").move_to(smem)
            smem_size = Text("~228 KB", font_size=10, color=C["label_c"],
                              font="Monospace").next_to(smem, DOWN, buff=0.03)

            regs = VGroup()
            for t in range(4):
                r = Rectangle(width=0.35, height=0.35, fill_color=C["reg_c"],
                                fill_opacity=0.4, stroke_color=C["local_c"], stroke_width=1)
                r.move_to([sm_cx + (t - 1.5) * 0.42, sm_top_y - 1.5, 0])
                rl = Text(f"R{t}", font_size=10, color=C["fg"], font="Monospace").move_to(r)
                regs.add(VGroup(r, rl))

            reg_label = Text("Registers (per-thread)", font_size=10,
                              color=C["label_c"], font="Monospace")
            reg_label.move_to([sm_cx, sm_top_y - 1.9, 0])

            self.add(sm_box, sm_title, smem, smem_lbl, smem_size, regs, reg_label)
            if sm_idx == 1:
                shared_target = smem.get_right()
                local_target = regs.get_right()

        # Croqtile side (right)
        crk_cx = 4.0
        crk_label = Text("Croqtile Specifiers", font_size=16, color=C["label_c"], font="Monospace")
        crk_label.move_to([crk_cx, 2.5, 0])
        self.add(crk_label)

        spec_data = [
            ("=> global", C["global_c"], 1.8, "Full device memory\nAll threads, all blocks"),
            ("=> shared", C["shared_c"], 0.0, "Block-scoped SRAM\nAll threads in one block"),
            ("=> local", C["local_c"], -1.2, "Thread-private\nRegisters / local scratch"),
        ]

        spec_boxes = {}
        for syntax, color, y, desc in spec_data:
            box = Rectangle(width=3.0, height=0.9, fill_color=color,
                            fill_opacity=0.15, stroke_color=color, stroke_width=2)
            box.move_to([crk_cx, y, 0])

            name_t = Text(syntax, font_size=14, color=color, font="Monospace")
            name_t.move_to(box.get_top() + DOWN * 0.2)

            desc_t = Text(desc, font_size=10, color=C["label_c"], font="Monospace",
                           line_spacing=1.2)
            desc_t.move_to(box.get_bottom() + UP * 0.22)

            self.add(box, name_t, desc_t)
            spec_boxes[syntax] = box

        # Arrows from Croqtile specifiers to concrete memory boxes
        arr_global = Arrow(spec_boxes["=> global"].get_left(),
                           dram.get_right(),
                           buff=0.1, stroke_width=2, color=C["global_c"],
                           max_tip_length_to_length_ratio=0.05)
        arr_shared = Arrow(spec_boxes["=> shared"].get_left(),
                           shared_target,
                           buff=0.08, stroke_width=2, color=C["shared_c"],
                           max_tip_length_to_length_ratio=0.05)
        arr_local = Arrow(spec_boxes["=> local"].get_left(),
                          local_target,
                          buff=0.08, stroke_width=2, color=C["local_c"],
                          max_tip_length_to_length_ratio=0.05)
        self.add(arr_global, arr_shared, arr_local)

        # Speed/size annotation along far left
        far_left = -6.5
        top_mark = 1.8
        bot_mark = sm_top_y - 1.9
        speed_arr = Arrow([far_left, top_mark, 0], [far_left, bot_mark, 0],
                          buff=0.05, stroke_width=1.5, color=C["fg3"],
                          max_tip_length_to_length_ratio=0.06)
        slow_lbl = Text("slower\nlarger", font_size=9, color=C["fg3"],
                          font="Monospace", line_spacing=1.1)
        slow_lbl.move_to([far_left + 0.55, top_mark - 0.3, 0])
        fast_lbl = Text("faster\nsmaller", font_size=9, color=C["fg3"],
                         font="Monospace", line_spacing=1.1)
        fast_lbl.move_to([far_left + 0.55, bot_mark + 0.3, 0])
        self.add(speed_arr, fast_lbl, slow_lbl)
