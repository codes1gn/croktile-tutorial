"""
Figure 4: Shared Memory Reuse — why => shared matters.
Left: without shared (each thread loads its own copy from global).
Right: with shared (one DMA fills shared, all threads read from it).
Tile A is shown as the same-size box everywhere to emphasize it's the same data.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from theme import parse_theme
from manim import *

C, THEME = parse_theme()

TILE_W = 1.0
TILE_H = 0.5


class SharedReuse(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text("Data Reuse: local vs shared", font_size=24,
                      color=C["fg"], font="Monospace")
        title.to_edge(UP, buff=0.3)
        self.add(title)

        sep = DashedLine(UP * 2.5, DOWN * 3.5, color=C["dim"], dash_length=0.1)
        self.add(sep)

        # --- Left: => local (no reuse) ---
        lt = Text("=> local (no reuse)", font_size=16, color=C["local_c"],
                   font="Monospace")
        lt.move_to(LEFT * 3.5 + UP * 2.1)
        self.add(lt)

        lx = LEFT * 3.5

        # Global Memory container (left)
        glob_l = Rectangle(width=5.0, height=1.2, fill_color=C["global_c"],
                            fill_opacity=0.08, stroke_color=C["global_c"], stroke_width=1.5)
        glob_l.move_to(lx + UP * 1.2)
        gl_label = Text("Global Memory", font_size=11, color=C["global_c"],
                         font="Monospace")
        gl_label.move_to(glob_l.get_top() + DOWN * 0.18)
        self.add(glob_l, gl_label)

        # Tile A inside global memory (left)
        tile_gl = Rectangle(width=TILE_W, height=TILE_H, fill_color=C["global_c"],
                             fill_opacity=0.25, stroke_color=C["global_c"], stroke_width=1.5)
        tile_gl.move_to(lx + UP * 0.95)
        tile_gl_t = Text("tile A", font_size=11, color=C["global_c"],
                          font="Monospace").move_to(tile_gl)
        self.add(tile_gl, tile_gl_t)

        # 4 local memory containers, each with a copy of tile A
        LOCAL_DY = 0.4
        for i in range(4):
            tx = lx + LEFT * 1.9 + RIGHT * i * 1.26

            # Local Memory container
            loc_box = Rectangle(width=TILE_W + 0.2, height=TILE_H + 0.45,
                                fill_color=C["local_c"], fill_opacity=0.06,
                                stroke_color=C["local_c"], stroke_width=1)
            loc_box.move_to(tx + DOWN * (0.3 + LOCAL_DY))
            loc_label = Text(f"local_{i}", font_size=9, color=C["local_c"],
                              font="Monospace")
            loc_label.move_to(loc_box.get_top() + DOWN * 0.13)
            self.add(loc_box, loc_label)

            # Same-size tile A copy inside local
            tile_copy = Rectangle(width=TILE_W, height=TILE_H,
                                  fill_color=C["local_c"], fill_opacity=0.2,
                                  stroke_color=C["local_c"], stroke_width=1)
            tile_copy.move_to(tx + DOWN * (0.45 + LOCAL_DY))
            tc_t = Text("tile A", font_size=9, color=C["local_c"],
                         font="Monospace").move_to(tile_copy)
            self.add(tile_copy, tc_t)

            # Arrow: global tile A -> local copy
            a1 = Arrow(tile_gl.get_bottom() + RIGHT * (i - 1.5) * 0.25,
                       loc_box.get_top(), buff=0.05, stroke_width=1.2,
                       color=C["global_c"],
                       max_tip_length_to_length_ratio=0.12)
            self.add(a1)

            # Thread circle below local
            tc = Circle(radius=0.18, fill_color=C["green"], fill_opacity=0.6,
                        stroke_color=C["fg"], stroke_width=1)
            tc.move_to(tx + DOWN * (1.3 + LOCAL_DY))
            tt = Text(f"t{i}", font_size=10, color=C["fg"], font="Monospace").move_to(tc)
            a2 = Arrow(loc_box.get_bottom(), tc.get_top(), buff=0.05,
                       stroke_width=1.2, color=C["local_c"],
                       max_tip_length_to_length_ratio=0.3)
            self.add(tc, tt, a2)

        cost_l = Text("4 copies of the same tile A\n4× bandwidth", font_size=11,
                       color=C["red"], font="Monospace")
        cost_l.move_to(lx + DOWN * (2.3 + LOCAL_DY))
        self.add(cost_l)

        # --- Right: => shared (reuse) ---
        rt_title = Text("=> shared (reuse)", font_size=16, color=C["shared_c"],
                         font="Monospace")
        rt_title.move_to(RIGHT * 3.5 + UP * 2.1)
        self.add(rt_title)

        rx = RIGHT * 3.5

        # Global Memory container (right)
        glob_r = Rectangle(width=5.0, height=1.2, fill_color=C["global_c"],
                            fill_opacity=0.08, stroke_color=C["global_c"], stroke_width=1.5)
        glob_r.move_to(rx + UP * 1.2)
        gr_label = Text("Global Memory", font_size=11, color=C["global_c"],
                         font="Monospace")
        gr_label.move_to(glob_r.get_top() + DOWN * 0.18)
        self.add(glob_r, gr_label)

        # Tile A inside global memory (right)
        tile_gr = Rectangle(width=TILE_W, height=TILE_H, fill_color=C["global_c"],
                             fill_opacity=0.25, stroke_color=C["global_c"], stroke_width=1.5)
        tile_gr.move_to(rx + UP * 0.95)
        tile_gr_t = Text("tile A", font_size=11, color=C["global_c"],
                          font="Monospace").move_to(tile_gr)
        self.add(tile_gr, tile_gr_t)

        # Shared Memory container with one copy of tile A
        SHARED_DY = 0.4
        smem_box = Rectangle(width=TILE_W + 0.6, height=TILE_H + 0.55,
                              fill_color=C["shared_c"], fill_opacity=0.08,
                              stroke_color=C["shared_c"], stroke_width=2)
        smem_box.move_to(rx + DOWN * (0.2 + SHARED_DY))
        smem_label = Text("Shared Memory", font_size=11, color=C["shared_c"],
                           font="Monospace")
        smem_label.move_to(smem_box.get_top() + DOWN * 0.15)
        self.add(smem_box, smem_label)

        # Same-size tile A inside shared memory
        tile_sm = Rectangle(width=TILE_W, height=TILE_H, fill_color=C["shared_c"],
                             fill_opacity=0.25, stroke_color=C["shared_c"], stroke_width=1.5)
        tile_sm.move_to(rx + DOWN * (0.35 + SHARED_DY))
        tile_sm_t = Text("tile A", font_size=11, color=C["shared_c"],
                          font="Monospace").move_to(tile_sm)
        self.add(tile_sm, tile_sm_t)

        # DMA arrow: global -> shared
        dma = Arrow(tile_gr.get_bottom(), smem_box.get_top(), buff=0.05,
                    stroke_width=2, color=C["arrow"],
                    max_tip_length_to_length_ratio=0.15)
        dma_lbl = Text("1× dma.copy", font_size=11, color=C["arrow"],
                        font="Monospace")
        dma_lbl.next_to(dma, RIGHT, buff=0.1)
        self.add(dma, dma_lbl)

        # 4 threads reading from shared
        for i in range(4):
            tx = rx + LEFT * 1.3 + RIGHT * i * 0.9
            tc = Circle(radius=0.18, fill_color=C["green"], fill_opacity=0.6,
                        stroke_color=C["fg"], stroke_width=1)
            tc.move_to(tx + DOWN * (1.3 + SHARED_DY))
            tt = Text(f"t{i}", font_size=10, color=C["fg"], font="Monospace").move_to(tc)

            a = Arrow(smem_box.get_bottom() + RIGHT * (i - 1.5) * 0.5,
                      tc.get_top(), buff=0.05, stroke_width=1.2,
                      color=C["shared_c"],
                      max_tip_length_to_length_ratio=0.2)
            self.add(tc, tt, a)

        cost_r = Text("1 copy, all threads read it\n1× bandwidth", font_size=11,
                       color=C["green"], font="Monospace")
        cost_r.move_to(rx + DOWN * (2.3 + SHARED_DY))
        self.add(cost_r)
