# Development gate

Combined pass: `False`. Outer test unused. Confirmation **refuses** unless this is true on every warm stage of every domain.

## UCI pass=`False`

- **S1**: Hybrid AP=0.8106 vs baseline 0.7694 Δ=0.0412 margin=0.0231 pos=True mat=True cold=None
- **S2**: Hybrid AP=0.9128 vs baseline 0.9067 Δ=0.0062 margin=0.0100 pos=True mat=False cold=None
- **S0**: Hybrid AP=0.4612 vs baseline 0.5010 Δ=-0.0398 margin=0.0500 pos=None mat=None cold=True

## OULAD pass=`False`

- **35pct**: Hybrid AP=0.8089 vs baseline 0.8087 Δ=0.0002 margin=0.0191 pos=True mat=False cold=None
- **50pct**: Hybrid AP=0.8576 vs baseline 0.8563 Δ=0.0013 margin=0.0144 pos=True mat=False cold=None
- **75pct**: Hybrid AP=0.8969 vs baseline 0.8989 Δ=-0.0020 margin=0.0101 pos=False mat=False cold=None
- **100pct**: Hybrid AP=0.9230 vs baseline 0.9260 Δ=-0.0029 margin=0.0100 pos=False mat=False cold=None
- **20pct**: Hybrid AP=0.7609 vs baseline 0.7684 Δ=-0.0075 margin=0.0200 pos=None mat=None cold=True
