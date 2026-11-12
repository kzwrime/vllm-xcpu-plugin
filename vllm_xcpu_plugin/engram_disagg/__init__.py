"""DeepSeek-V4.1 Engram disaggregation (attention ranks + Engram ranks).

The Engram hash tables are served by dedicated MPI ranks that follow the
attention ranks in one MPMD world (the AF-EP shape). Attention ranks send
the hash ids of their tokens once per forward and decode the rows the Engram
ranks return; the Engram ranks spend their steady state inside one native
service loop (``torch_xcpu.ops.deepseek_v41_engram_serve``).

Role-neutral pieces live in :mod:`engram_disagg.common`; the attention-side
client in :mod:`engram_disagg.attn`; the Engram-rank process in
:mod:`engram_disagg.server`.
"""
