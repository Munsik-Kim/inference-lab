# P1 amendment 01 — boundary pair rather than a changing lower bulk

This amendment was written before the first P1 primary integration on
2026-10-23 KST. Its exact bytes and UTC freeze time are recorded in the P1
execution freeze. P0's 17 primary observations were already known; P1 primary
outputs observed at this design stage: zero.

P0 `theory.md` described a candidate changing the upper endpoint of the lower
32 singular values. That candidate would also change the lower bulk spectrum.
The P1 design instead fixes the measured singular bases of the saved P0 block-C
target C0, and changes only singular values 32 and 33:

    sigma32 = 0.6 + g/2, sigma33 = 0.6 - g/2,
    g in {0.4, 0.1, 0.025}.

The other 62 singular values and all singular bases remain fixed within the
predeclared FP64 checks. g=.4 preserves C0's array bytes exactly. Other targets
are rank-two updates in that fixed basis. Measured spectra, basis overlaps,
norms, condition and individual rank-32 loss floors are retained.

The two boundary amplitudes and target norm/loss floor also change. This is a
boundary-pair comparison, not a claim to change gap while keeping all magnitude
quantities constant. P0 documents, candidate text, source, data and receipts
remain byte-identical. The new seed, worst-angle metric and longer horizon
belong to P1. All numerical acceptance criteria and the 12 matrix reference
selections are frozen before the P1 matrix is observed.
