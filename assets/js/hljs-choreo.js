hljs.registerLanguage('choreo', function(hljs) {
  var CHOREO_TYPES =
    'void bool int half bfp16 float double ' +
    'tf32 f64 f32 f16 bf16 f8 f8_e4m3 f8_e5m2 f8_ue4m3 f8_ue8m0 ' +
    'f6_e3m2 f6_e2m3 f4_e2m1 ' +
    'u64 s64 u32 s32 u16 s16 u8 s8 u6 s6 u4 s4 u2 s2 u1 bin1 ' +
    'half8 auto stream';

  var CHOREO_KEYWORDS =
    '__co__ __cok__ __real_cok__ __co_device__ __device__ __cpp__ ' +
    'parallel foreach with in by where after ' +
    'if else while break continue return ' +
    'inthreads call select cdiv ' +
    'vectorize event';

  var CHOREO_STORAGE =
    'local shared global mutable';

  var CHOREO_PARALLEL_LEVELS =
    'block group group-4 thread device term';

  var CHOREO_BUILTINS =
    'dma tma mma sync trigger assert swap rotate print println frag ' +
    'copy transp pad swizzle swiz sp zfill any async ' +
    'span span_as mdata data view from chunkat chunk subspan modspan step stride at ' +
    'fill load store row col scale commit ' +
    'wait';

  var CHOREO_MATH =
    '__acos __asin __atan __atan2 __ceil __cos __cosh __exp __expm1 ' +
    '__floor __gelu __isfinite __round __rsqrt __sigmoid __sinh ' +
    '__softplus __sqrt __tan __log1p __log __pow __sign __sin __tanh ' +
    '__alignup __aligndown';

  return {
    name: 'Choreo',
    aliases: ['choreo', 'co'],
    case_insensitive: false,
    keywords: {
      keyword: CHOREO_KEYWORDS,
      type: CHOREO_TYPES,
      built_in: CHOREO_BUILTINS + ' ' + CHOREO_MATH,
      literal: 'true false',
    },
    contains: [
      hljs.C_LINE_COMMENT_MODE,
      hljs.C_BLOCK_COMMENT_MODE,
      hljs.QUOTE_STRING_MODE,
      hljs.C_NUMBER_MODE,
      {
        className: 'meta',
        begin: /#\s*(define|if|ifdef|ifndef|else|elif|endif|error|pragma|include)\b/,
        end: /$/,
        contains: [
          { begin: /\\\n/, relevance: 0 },
          hljs.QUOTE_STRING_MODE,
          hljs.C_LINE_COMMENT_MODE,
        ]
      },
      {
        className: 'keyword',
        begin: /\b(parallel\.async|inthreads\.async|dma\.copy\.async|dma\.copy\.swiz|tma\.copy\.async|tma\.copy\.swiz|mma\.fill\.f16|mma\.fill\.f32|mma\.load\.swiz|mma\.store\.transp|mma\.row\.row|mma\.row\.col|mma\.col\.row|mma\.col\.col|mma\.row\.row\.sp|mma\.row\.row\.scale|dma\.copy|dma\.transp|dma\.any|tma\.copy|mma\.fill|mma\.load|mma\.store|mma\.commit|mma\.scale|sync\.shared|sync\.global|sync\.local)\b/
      },
      {
        className: 'title',
        begin: /\b__co__\s+/,
        end: /\(/,
        excludeEnd: true,
        contains: [
          { className: 'type', begin: /\b(void|s32|f16|f32|bf16|f8_e4m3|auto)\b/ },
          { className: 'title', begin: /\b[a-zA-Z_]\w*\b/ },
        ]
      },
      {
        className: 'built_in',
        begin: /\.(copy|transp|pad|swizzle|swiz|sp|zfill|any|async|span|span_as|mdata|data|view|from|chunkat|chunk|subspan|modspan|step|stride|at|fill|load|store|row|col|scale|commit)\b/
      },
      {
        className: 'variable',
        begin: /\b(block|group|group-4|thread|device|term)\b/,
        relevance: 0
      },
      {
        className: 'type',
        begin: /\b(local|shared|global)\b/
      },
      {
        className: 'symbol',
        begin: /=>/
      },
      {
        className: 'meta',
        begin: /\b__cpp__\s*\(\s*R"\(/,
        end: /\)"\s*\)\s*;/,
      },
      {
        className: 'meta',
        begin: /\b__cpp__\s*\(\s*"/,
        end: /"\s*\)\s*;/,
      },
    ]
  };
});
