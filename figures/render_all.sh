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

echo "=== Ch04 static ==="
mkdir -p "$DOCS/images/ch04"
render_static ch04_fig1_tensor_contraction.py TensorContraction fig1_tensor_contraction "$DOCS/images/ch04"
render_static ch04_fig2_register_loading.py RegisterLoading fig2_register_loading "$DOCS/images/ch04"
render_static ch04_fig3_mma_syntax.py MMASyntax fig3_mma_syntax "$DOCS/images/ch04"
render_static ch04_fig2_sm86_vs_sm90.py SM86vsSM90 fig4_sm86_vs_sm90 "$DOCS/images/ch04"

echo "=== Ch05 static ==="
mkdir -p "$DOCS/images/ch05"
render_static ch05_fig1_role_split.py Ch05Fig1RoleSplit fig1_role_split "$DOCS/images/ch05"
render_static ch05_fig2_persistent_kernel.py Ch05Fig2PersistentKernel fig2_persistent_kernel "$DOCS/images/ch05"

echo "=== Ch06 static ==="
mkdir -p "$DOCS/images/ch06"
render_static ch06_fig1_pipeline_timeline.py PipelineTimeline fig1_pipeline_timeline "$DOCS/images/ch06"
render_static ch06_fig2_event_credit_flow.py EventCreditFlow fig2_event_credit_flow "$DOCS/images/ch06"

echo "=== Ch07 static ==="
mkdir -p "$DOCS/images/ch07"
render_static ch07_fig1_tma_vs_dma.py TmaVsDma fig1_tma_vs_dma "$DOCS/images/ch07"
render_static ch07_fig2_swizzle.py SwizzleBanks fig2_swizzle "$DOCS/images/ch07"

echo "=== Ch08 ==="
mkdir -p "$DOCS/images/ch08"
render_static ch08_fig1_escape_hatch.py Ch08Fig1EscapeHatch fig1_escape_hatch "$DOCS/images/ch08"

echo "=== Ch09 ==="
mkdir -p "$DOCS/images/ch09"
render_static ch09_fig1_debug_workflow.py DebugWorkflow fig1_debug_workflow "$DOCS/images/ch09"

echo "=== Ch02 animations ==="
render_video ch02_anim1_element_vs_block.py ElementVsBlockAnim anim1_element_vs_block "$DOCS/videos/ch02"
render_video ch02_anim2_tiled_add.py TiledAddAnim anim2_tiled_add "$DOCS/videos/ch02"
render_video ch02_anim3_chunkat.py ChunkatAnim anim3_chunkat "$DOCS/videos/ch02"

echo "=== Ch03 static ==="
mkdir -p "$DOCS/images/ch03"
render_static ch03_fig1_virtual_parallelism.py VirtualParallelism fig1_virtual_parallelism "$DOCS/images/ch03"
render_static ch03_fig2_logical_vs_physical.py LogicalVsPhysical fig2_logical_vs_physical "$DOCS/images/ch03"
render_static ch03_fig3_space_specifiers.py SpaceSpecifiers fig3_space_specifiers "$DOCS/images/ch03"
render_static ch03_fig4_shared_reuse.py SharedReuse fig4_shared_reuse "$DOCS/images/ch03"
render_static ch03_fig5_scalar_matmul.py ScalarMatmul fig5_scalar_matmul "$DOCS/images/ch03"
render_static ch03_fig6_dma_matmul.py DmaMatmul fig6_dma_matmul "$DOCS/images/ch03"
render_static ch03_fig7_matmul_gpu_layout.py MatmulGpuLayout fig7_matmul_gpu_layout "$DOCS/images/ch03"

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
