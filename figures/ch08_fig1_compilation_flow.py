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
            ".co \u2192 croqtile compiler \u2192 .cu \u2192 nvcc \u2192 binary",
            font_size=24, color=C["fg"], font="Monospace",
        )
        title.to_edge(UP, buff=0.22)
        self.add(title)

        left_x = -4.0

        src_lbl = Text("kernel.co", font_size=20, color=C["fg"], font="Monospace")
        src_lbl.move_to(RIGHT * left_x + UP * 2.35)
        self.add(src_lbl)

        def code_box(lines, y, col, tag, w=4.0, h=None):
            if h is None:
                h = 0.32 * len(lines) + 0.35
            r = RoundedRectangle(
                width=w, height=h, corner_radius=0.1,
                fill_color=col, fill_opacity=0.12,
                stroke_color=col, stroke_width=2,
            )
            r.move_to(RIGHT * left_x + UP * y)
            self.add(r)
            tag_t = Text(tag, font_size=12, color=col, font="Monospace")
            tag_t.move_to(r.get_corner(UR) + LEFT * 0.7 + DOWN * 0.14)
            self.add(tag_t)
            for i, ln in enumerate(lines):
                t = Text(ln, font_size=11, color=C["fg2"], font="Monospace")
                t.move_to(r.get_top() + DOWN * (0.25 + i * 0.28))
                t.align_to(r.get_left() + RIGHT * 0.18, LEFT)
                self.add(t)
            return r

        b1 = code_box(
            ["__device__ float fast_rsqrt(x)",
             "  { return __frsqrt_rn(x); }"],
            1.55, C["purple"], "\u2460 pass-through",
        )
        b2 = code_box(
            ["__co__ void kernel(f32 [M,N] in) {",
             "  parallel ... : block { dma, mma }",
             '  __cpp__("asm volatile(...)");',
             "}"],
            0.4, C["green"], "\u2461 transform + \u2462 splice",
        )
        b3 = code_box(
            ["int main() {",
             "  auto buf = choreo::make_spandata();",
             "  kernel(buf.view());",
             "}"],
            -0.75, C["blue"], "\u2463 pass-through",
        )

        mid_x = 0.6
        compiler = RoundedRectangle(
            width=2.4, height=0.7, corner_radius=0.12,
            fill_color=C["green"], fill_opacity=0.18,
            stroke_color=C["green"], stroke_width=2.5,
        )
        compiler.move_to(RIGHT * mid_x + UP * 0.4)
        comp_lbl = Text("croqtile", font_size=18, color=C["green"], font="Monospace")
        comp_sub = Text("compiler", font_size=14, color=C["fg3"], font="Monospace")
        comp_lbl.move_to(compiler.get_center() + UP * 0.12)
        comp_sub.move_to(compiler.get_center() + DOWN * 0.16)
        self.add(compiler, comp_lbl, comp_sub)

        arr_in = Arrow(
            b2.get_right(), compiler.get_left(),
            buff=0.1, stroke_width=2.5, color=C["arrow"],
            max_tip_length_to_length_ratio=0.14,
        )
        self.add(arr_in)

        right_x = 4.0
        cu_lbl = Text("generated .cu", font_size=18, color=C["fg"], font="Monospace")
        cu_lbl.move_to(RIGHT * right_x + UP * 2.35)
        self.add(cu_lbl)

        def out_box(label, sub, y, col, h=0.6):
            r = RoundedRectangle(
                width=3.6, height=h, corner_radius=0.1,
                fill_color=col, fill_opacity=0.1,
                stroke_color=col, stroke_width=2,
            )
            r.move_to(RIGHT * right_x + UP * y)
            self.add(r)
            t = Text(label, font_size=13, color=col, font="Monospace")
            t.move_to(r.get_center() + UP * 0.1)
            self.add(t)
            s = Text(sub, font_size=11, color=C["dim"], font="Monospace")
            s.move_to(r.get_center() + DOWN * 0.14)
            self.add(s)
            return r

        o1 = out_box("__device__ fast_rsqrt", "copied verbatim", 1.65, C["purple"])
        o2 = out_box("__global__ __choreo_kernel", "generated from __co__", 0.85, C["green"])
        o3 = out_box('asm volatile("setmaxnreg")', "spliced from __cpp__", 0.15, C["orange"])
        o4 = out_box("int main() { ... }", "host launch wrapper", -0.55, C["blue"])

        for o in [o1, o2, o3]:
            a = Arrow(
                compiler.get_right(), o.get_left(),
                buff=0.08, stroke_width=2, color=C["arrow"],
                max_tip_length_to_length_ratio=0.12,
            )
            self.add(a)

        for src, dst in [(b1, o1), (b3, o4)]:
            a = Arrow(
                src.get_right(), dst.get_left(),
                buff=0.08, stroke_width=2, color=C["dim"],
                max_tip_length_to_length_ratio=0.12,
            )
            self.add(a)

        nvcc = RoundedRectangle(
            width=2.2, height=0.6, corner_radius=0.1,
            fill_color=C["fg3"], fill_opacity=0.15,
            stroke_color=C["fg3"], stroke_width=2,
        )
        nvcc.move_to(RIGHT * right_x + DOWN * 1.4)
        nvcc_lbl = Text("nvcc", font_size=18, color=C["fg"], font="Monospace")
        nvcc_lbl.move_to(nvcc.get_center())
        self.add(nvcc, nvcc_lbl)

        gpu_lbl = Text("GPU code", font_size=13, color=C["purple"], font="Monospace")
        gpu_lbl.move_to(RIGHT * (right_x - 0.9) + DOWN * 0.9)
        self.add(gpu_lbl)
        cpu_lbl = Text("CPU code", font_size=13, color=C["blue"], font="Monospace")
        cpu_lbl.move_to(RIGHT * (right_x + 0.9) + DOWN * 0.9)
        self.add(cpu_lbl)

        for o in [o1, o2, o3]:
            a = Arrow(
                o.get_bottom(), nvcc.get_top() + LEFT * 0.45,
                buff=0.06, stroke_width=1.8, color=C["purple"],
                max_tip_length_to_length_ratio=0.1,
            )
            self.add(a)
        a_host = Arrow(
            o4.get_bottom(), nvcc.get_top() + RIGHT * 0.45,
            buff=0.06, stroke_width=1.8, color=C["blue"],
            max_tip_length_to_length_ratio=0.1,
        )
        self.add(a_host)

        binary = RoundedRectangle(
            width=2.0, height=0.5, corner_radius=0.1,
            fill_color=C["green"], fill_opacity=0.2,
            stroke_color=C["green"], stroke_width=2,
        )
        binary.move_to(RIGHT * right_x + DOWN * 2.25)
        bin_lbl = Text("GPU binary", font_size=15, color=C["green"], font="Monospace")
        bin_lbl.move_to(binary.get_center())
        self.add(binary, bin_lbl)

        a_bin = Arrow(
            nvcc.get_bottom(), binary.get_top(),
            buff=0.06, stroke_width=2.5, color=C["arrow"],
            max_tip_length_to_length_ratio=0.12,
        )
        self.add(a_bin)

        foot = Text(
            "__device__ \u2192 GPU verbatim  |  __co__ \u2192 GPU transformed  |"
            "  host C++ \u2192 CPU  |  __cpp__ \u2192 spliced",
            font_size=12, color=C["dim"], font="Monospace",
        )
        foot.to_edge(DOWN, buff=0.2)
        self.add(foot)
