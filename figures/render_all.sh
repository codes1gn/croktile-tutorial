#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

DOCS="../docs/assets"
mkdir -p "$DOCS/images/ch01" "$DOCS/images/ch02" "$DOCS/videos/ch01" "$DOCS/videos/ch02"
mkdir -p "$DOCS/images/optimization"

render_static() {
    local script="$1" scene="$2" outname="$3" dest_dir="$4"
    for theme in dark light; do
        echo "  [$theme] $script::$scene → ${outname}_${theme}.png"
        MANIM_THEME="$theme" python3 -m manim render -ql --format=png \
            -o "${outname}_${theme}" \
            "$script" "$scene" 2>&1 | tail -1
        # find the output
        local found=$(find media -name "${outname}_${theme}.png" 2>/dev/null | head -1)
        if [ -n "$found" ]; then
            cp "$found" "$dest_dir/${outname}_${theme}.png"
            echo "    → copied to $dest_dir/${outname}_${theme}.png"
        else
            echo "    WARN: output not found"
        fi
    done
}

render_video() {
    local script="$1" scene="$2" outname="$3" dest_dir="$4"
    for theme in dark light; do
        echo "  [$theme] $script::$scene → ${outname}_${theme}.mp4"
        MANIM_THEME="$theme" python3 -m manim render -ql \
            -o "${outname}_${theme}" \
            "$script" "$scene" 2>&1 | tail -1
        local found=$(find media -name "${outname}_${theme}.mp4" 2>/dev/null | head -1)
        if [ -n "$found" ]; then
            cp "$found" "$dest_dir/${outname}_${theme}.mp4"
            echo "    → copied to $dest_dir/${outname}_${theme}.mp4"
        else
            echo "    WARN: output not found"
        fi
    done
}

echo "=== Ch01 ==="
render_static ch01_compile_anim.py CompileAndRun compile_and_run "$DOCS/images/ch01"

echo "=== Ch02 static ==="
render_static ch02_fig1_element_vs_block.py ElementVsBlock fig1_element_vs_block "$DOCS/images/ch02"
render_static ch02_fig2_tiled_add.py TiledAdd fig2_tiled_add "$DOCS/images/ch02"
render_static ch02_fig3_chunkat.py ChunkatSemantics fig3_chunkat "$DOCS/images/ch02"
render_static ch02_fig_compose.py ComposeOperator fig_compose "$DOCS/images/ch02"
render_static ch02_fig_dma_copy.py DmaCopy fig_dma_copy "$DOCS/images/ch02"
render_static ch02_fig_extent.py ExtentOperator fig_extent "$DOCS/images/ch02"
render_static ch02_fig_future_data.py FutureData fig_future_data "$DOCS/images/ch02"
render_static ch02_fig_memory_hierarchy.py MemoryHierarchy fig_memory_hierarchy "$DOCS/images/ch02"
render_static ch02_fig_span.py SpanDimension fig_span "$DOCS/images/ch02"

echo "=== Ch02 animations ==="
render_video ch02_anim1_element_vs_block.py ElementVsBlockAnim anim1_element_vs_block "$DOCS/videos/ch02"
render_video ch02_anim2_tiled_add.py TiledAddAnim anim2_tiled_add "$DOCS/videos/ch02"
render_video ch02_anim3_chunkat.py ChunkatAnim anim3_chunkat "$DOCS/videos/ch02"

echo "=== Optimization ==="
for scene in BaselineKernel Step2ThreeStage SplitOutput1p2c OccupancyCliff; do
    render_static optimization/dense_gemm_figures.py "$scene" "${scene}_ManimCE_v0.19.1" "$DOCS/images/optimization"
done
for scene in SparsityPattern MetadataBottleneck ThreeStagePipelineJump; do
    render_static optimization/sparse_gemm_figures.py "$scene" "${scene}_ManimCE_v0.19.1" "$DOCS/images/optimization"
done
for scene in BlockScaleConcept TMAOverlap N256VsN128 OptimizationLadder; do
    render_static optimization/blockscale_gemm_figures.py "$scene" "${scene}_ManimCE_v0.19.1" "$DOCS/images/optimization"
done

echo "=== Done ==="
echo "Rendered files:"
find "$DOCS/images" -name "*_dark.png" -o -name "*_light.png" | sort
find "$DOCS/videos" -name "*_dark.mp4" -o -name "*_light.mp4" | sort
