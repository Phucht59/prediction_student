# V6.1 errata and interpretive addendum

The checksum-protected V6.1 artifact and report are not edited. The blank H5
evidence sentence is clarified here:

**H5 data limitation: PARTIAL.** Parameter matching, dilation correction, a
direct CNN skip, and a parallel CNN/BiLSTM candidate did not produce a stable
development-gate improvement over the full serial control. Temporal-order
destruction reduced Macro-F1, so chronology contains signal, but the incremental
local CNN signal beyond the BiLSTM and aggregate/static representation remained
small. This supports a bounded data/representation explanation, not a claim that
OULAD contains no temporal information.
