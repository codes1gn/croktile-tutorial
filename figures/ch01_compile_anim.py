"""
Ch1: Terminal compile-and-run animation.
Simulates typing commands and showing output in a terminal.
"""
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from theme import parse_theme

from manim import *

C, THEME = parse_theme()


class CompileAndRun(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        term_box = Rectangle(width=11, height=6.5, fill_color=C["term_bg"],
                             fill_opacity=1, stroke_color=C["term_border"], stroke_width=2)
        term_box.move_to(ORIGIN)

        title_bar = Rectangle(width=11, height=0.45, fill_color=C["term_bar"],
                              fill_opacity=1, stroke_width=0)
        title_bar.move_to(term_box.get_top() + DOWN * 0.225)

        dots = VGroup()
        for i, c in enumerate(["#ff5f56", "#ffbd2e", "#27c93f"]):
            d = Circle(radius=0.08, fill_color=c, fill_opacity=1, stroke_width=0)
            d.move_to(title_bar.get_left() + RIGHT * (0.35 + i * 0.3))
            dots.add(d)

        title_text = Text("Terminal", font_size=13, color=C["output_c"], font="Monospace")
        title_text.move_to(title_bar)

        self.add(term_box, title_bar, dots, title_text)

        lines = VGroup()
        line_y = term_box.get_top()[1] - 0.85
        left_x = term_box.get_left()[0] + 0.4

        def add_line(text, color=C["cmd_c"], delay=0.03):
            nonlocal line_y
            t = Text(text, font_size=15, color=color, font="Monospace")
            t.move_to([left_x + t.width / 2, line_y, 0])
            lines.add(t)
            line_y -= 0.35
            return t

        def type_line(prompt_text, cmd_text, delay=0.03):
            nonlocal line_y
            prompt = Text(prompt_text, font_size=15, color=C["prompt_c"], font="Monospace")
            prompt.move_to([left_x + prompt.width / 2, line_y, 0])
            self.play(FadeIn(prompt), run_time=0.15)

            cmd = Text(cmd_text, font_size=15, color=C["cmd_c"], font="Monospace")
            cmd.move_to([left_x + prompt.width + 0.08 + cmd.width / 2, line_y, 0])

            cursor = Rectangle(width=0.1, height=0.22, fill_color=C["cursor_c"],
                                fill_opacity=0.8, stroke_width=0)
            cursor.move_to([left_x + prompt.width + 0.15, line_y, 0])
            self.add(cursor)

            for i, char in enumerate(cmd_text):
                partial = Text(cmd_text[:i + 1], font_size=15, color=C["cmd_c"], font="Monospace")
                partial.move_to([left_x + prompt.width + 0.08 + partial.width / 2, line_y, 0])
                cursor.move_to([left_x + prompt.width + 0.08 + partial.width + 0.06, line_y, 0])
                if i > 0:
                    self.remove(lines[-1])
                lines.add(partial)
                self.add(partial)
                self.wait(delay)

            self.remove(cursor)
            lines.add(prompt)
            line_y -= 0.35
            self.wait(0.15)

        # Scene 1: Show the file
        add_line("$ ls", C["prompt_c"])
        self.play(FadeIn(lines[-1]), run_time=0.2)
        self.wait(0.3)
        add_line("ele_add.co", C["output_c"])
        self.play(FadeIn(lines[-1]), run_time=0.15)
        self.wait(0.4)

        # Scene 2: Compile
        type_line("$ ", "croktile ele_add.co -o ele_add", delay=0.025)

        # Show compilation output
        self.wait(0.3)
        for msg in [
            "info: transpiling ele_add.co ...",
            "info: compiling CUDA kernel ...",
            "info: linking with nvcc ...",
        ]:
            add_line(msg, C["output_c"])
            self.play(FadeIn(lines[-1]), run_time=0.12)
            self.wait(0.2)

        add_line("info: build successful -> ele_add", C["success_c"])
        self.play(FadeIn(lines[-1]), run_time=0.15)
        self.wait(0.5)

        # Scene 3: Run
        type_line("$ ", "./ele_add", delay=0.035)
        self.wait(0.4)

        result = Text("Test Passed", font_size=17, color=C["success_c"], font="Monospace")
        result.move_to([left_x + result.width / 2, line_y, 0])
        line_y -= 0.35

        self.play(FadeIn(result), run_time=0.2)
        self.wait(0.3)

        check = Text("✓", font_size=22, color=C["success_c"])
        check.next_to(result, RIGHT, buff=0.2)
        self.play(FadeIn(check, scale=1.5), run_time=0.3)
        self.wait(0.5)

        # Scene 4: --help hint
        type_line("$ ", "croktile --help", delay=0.03)
        self.wait(0.3)

        help_lines = [
            "Usage: croktile [options] <input.co>",
            "",
            "Options:",
            "  -o <file>     Output binary name",
            "  -es           Emit source only (transpile)",
            "  -t <target>   Target platform (cuda, ...)",
            "  -v            Verbose compilation",
            "  --help        Show this help message",
        ]
        for h in help_lines:
            if h == "":
                line_y -= 0.15
                continue
            add_line(h, C["output_c"])
            self.play(FadeIn(lines[-1]), run_time=0.06)

        self.wait(2.5)
