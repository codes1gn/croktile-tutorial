"""
Ch08 Fig1 (combined): .co compilation flow with concrete code regions.
Shows which parts Croqtile transforms vs passes through, and how
__device__ (GPU) differs from host C++ (CPU) in the final nvcc split.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from theme import parse_theme
from manim import *

C, THEME = parse_theme()


class Ch08CompilationFlow(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text(
            ".co file → croqtile compiler → .cu intermediate → nvcc → binary",
            font_size=18, color=C["fg"], font="Monospace",
        )
        title.to_edge(UP, buff=0.25)
        self.add(title)

        # ── Left column: source .co regions ──
        left_x = -4.0
        src_lbl = Text("kernel.co", font_size=16, color=C["fg"], font="Monospace")
        src_lbl.move_to(RIGHT * left_x + UP * 2.3)
        self.add(src_lbl)

        def code_box(lines, y, col, tag, w=3.8, h=None):
            if h is None:
                h = 0.28 * len(lines) + 0.3
            r = RoundedRectangle(
                width=w, height=h, corner_radius=0.08,
                fill_color=col, fill_opacity=0.12,
                stroke_color=col, stroke_width=1.8,
            )
            r.move_to(RIGHT * left_x + UP * y)
            self.add(r)
            tag_t = Text(tag, font_size=10, color=col, font="Monospace")
            tag_t.move_to(r.get_corner(UR) + LEFT * 0.6 + DOWN * 0.12)
            self.add(tag_t)
            for i, ln in enumerate(lines):
                t = Text(ln, font_size=9, color=C["fg2"], font="Monospace")
                t.move_to(r.get_top() + DOWN * (0.22 + i * 0.24) + LEFT * 0.05)
                t.align_to(r.get_left() + RIGHT * 0.15, LEFT)
                self.add(t)
            return r

        b1 = code_box(
            ["__device__ float fast_rsqrt(float x)",
             "  { return __frsqrt_rn(x); }"],
            1.6, C["purple"], "① pass-through",
        )
        b2 = code_box(
            ["__co__ void my_kernel(f32 [M,N] in) {",
             "  parallel ... : block { dma, mma }",
             '  __cpp__("asm volatile(...)");',
             "}"],
            0.55, C["green"], "② transform + ③ splice",
        )
        b3 = code_box(
            ["int main() {",
             "  auto buf = choreo::make_spandata(...);",
             "  my_kernel(buf.view());",
             "}"],
            -0.55, C["blue"], "④ pass-through",
        )

        # ── Center: croqtile compiler ──
        mid_x = 0.6
        compiler = RoundedRectangle(
            width=2.2, height=0.65, corner_radius=0.1,
            fill_color=C["green"], fill_opacity=0.18,
            stroke_color=C["green"], stroke_width=2,
        )
        compiler.move_to(RIGHT * mid_x + UP * 0.55)
        comp_lbl = Text("croqtile", font_size=14, color=C["green"], font="Monospace")
        comp_sub = Text("compiler", font_size=11, color=C["fg3"], font="Monospace")
        comp_lbl.move_to(compiler.get_center() + UP * 0.1)
        comp_sub.move_to(compiler.get_center() + DOWN * 0.14)
        self.add(compiler, comp_lbl, comp_sub)

        arr_in = Arrow(
            b2.get_right(), compiler.get_left(),
            buff=0.1, stroke_width=2, color=C["arrow"],
            max_tip_length_to_length_ratio=0.15,
        )
        self.add(arr_in)

        # ── Right column: generated .cu regions ──
        right_x = 4.0
        cu_lbl = Text("__choreo_kernel.cu", font_size=14, color=C["fg"], font="Monospace")
        cu_lbl.move_to(RIGHT * right_x + UP * 2.3)
        self.add(cu_lbl)

        def out_box(label, sub, y, col, h=0.55):
            r = RoundedRectangle(
                width=3.4, height=h, corner_radius=0.08,
                fill_color=col, fill_opacity=0.1,
                stroke_color=col, stroke_width=1.5,
            )
            r.move_to(RIGHT * right_x + UP * y)
            self.add(r)
            t = Text(label, font_size=11, color=col, font="Monospace")
            t.move_to(r.get_center() + UP * 0.08)
            self.add(t)
            s = Text(sub, font_size=9, color=C["dim"], font="Monospace")
            s.move_to(r.get_center() + DOWN * 0.13)
            self.add(s)
            return r

        o1 = out_box("__device__ fast_rsqrt", "copied verbatim", 1.7, C["purple"])
        o2 = out_box("__global__ __choreo_my_kernel", "generated from __co__", 0.95, C["green"])
        o3 = out_box('asm volatile("setmaxnreg...")', "spliced from __cpp__", 0.3, C["orange"])
        o4 = out_box("int main() { ... }", "host launch wrapper", -0.35, C["blue"])

        # Arrows: compiler -> outputs
        for o in [o1, o2, o3]:
            a = Arrow(
                compiler.get_right(), o.get_left(),
                buff=0.08, stroke_width=1.5, color=C["arrow"],
                max_tip_length_to_length_ratio=0.12,
            )
            self.add(a)

        # pass-through arrows (device + host)
        for src, dst in [(b1, o1), (b3, o4)]:
            a = Arrow(
                src.get_right(), dst.get_left(),
                buff=0.08, stroke_width=1.5, color=C["dim"],
                max_tip_length_to_length_ratio=0.12,
            )
            self.add(a)

        # ── Bottom: nvcc split ──
        nvcc = RoundedRectangle(
            width=2.0, height=0.55, corner_radius=0.1,
            fill_color=C["fg3"], fill_opacity=0.15,
            stroke_color=C["fg3"], stroke_width=1.8,
        )
        nvcc.move_to(RIGHT * right_x + DOWN * 1.2)
        nvcc_lbl = Text("nvcc", font_size=14, color=C["fg"], font="Monospace")
        nvcc_lbl.move_to(nvcc.get_center())
        self.add(nvcc, nvcc_lbl)

        # GPU side label
        gpu_lbl = Text("GPU code", font_size=10, color=C["purple"], font="Monospace")
        gpu_lbl.move_to(RIGHT * (right_x - 0.8) + DOWN * 0.75)
        self.add(gpu_lbl)
        # CPU side label
        cpu_lbl = Text("CPU code", font_size=10, color=C["blue"], font="Monospace")
        cpu_lbl.move_to(RIGHT * (right_x + 0.8) + DOWN * 0.75)
        self.add(cpu_lbl)

        # Arrows to nvcc
        for o in [o1, o2, o3]:
            a = Arrow(
                o.get_bottom(), nvcc.get_top() + LEFT * 0.4,
                buff=0.06, stroke_width=1.2, color=C["purple"],
                max_tip_length_to_length_ratio=0.1,
            )
            self.add(a)
        a_host = Arrow(
            o4.get_bottom(), nvcc.get_top() + RIGHT * 0.4,
            buff=0.06, stroke_width=1.2, color=C["blue"],
            max_tip_length_to_length_ratio=0.1,
        )
        self.add(a_host)

        # Binary
        binary = RoundedRectangle(
            width=1.8, height=0.45, corner_radius=0.08,
            fill_color=C["green"], fill_opacity=0.2,
            stroke_color=C["green"], stroke_width=1.8,
        )
        binary.move_to(RIGHT * right_x + DOWN * 2.1)
        bin_lbl = Text("GPU binary", font_size=12, color=C["green"], font="Monospace")
        bin_lbl.move_to(binary.get_center())
        self.add(binary, bin_lbl)

        a_bin = Arrow(
            nvcc.get_bottom(), binary.get_top(),
            buff=0.06, stroke_width=2, color=C["arrow"],
            max_tip_length_to_length_ratio=0.12,
        )
        self.add(a_bin)

        # Footer
        foot = Text(
            "__device__ → GPU verbatim  |  __co__ → GPU transformed  |  host C++ → CPU  |  __cpp__ → spliced into GPU",
            font_size=10, color=C["dim"], font="Monospace",
        )
        foot.to_edge(DOWN, buff=0.2)
        self.add(foot)
