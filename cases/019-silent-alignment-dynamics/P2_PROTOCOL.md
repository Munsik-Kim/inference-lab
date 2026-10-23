# Case019 P2 — checkpoint direction interventions, v1

Registered locally on 2026-10-23T04:51:36.785229+00:00 before any P2 primary ODE. This follows observed P0/P1 results. R2 was selected after seeing worst-half83.061701 and mode32 re-exit; this is not an unobserved-output preregistration. Initial checkpoint/intervention metrics are computed before primary freeze and explicitly disclosed.

Questions: (1) corroborate R2 tail/re-exit by finite-rho graph and additional physical RK4; (2) compare four directions with identical prediction singular values at each checkpoint; (3) separate immediate loss jump from subsequent learning.

## Fixed inputs and numerical model

P1 `results/p1_20261023-104046`, fixed target bases and R8 reused. R0/R2 mean manifest R_index0/2, streams[61023020,1,0/2], not ordinal numbering. General FP64 ODE L=.5||UV^T-C||F^2, Udot=(C-UV^T)V, Vdot=(C-UV^T)^TU. No1/d,1/m, clipping, jitter or conservation correction. Scaled A=U/sqrt(2rho),B=V/sqrt(2rho) starts from actual nonzero physical checkpoints. Primary DOP853 rtol1e-10/atol1e-12; reference original U/V RK4 .01/.005.

32 parents: g=.4/.025, rho1e-6, R0..7, t*=1/5. Four arms each,128 distinct integrations: CONTINUE(U,V), GAUGE(UO,VO), ALIGN oracle top target singular pairs, ROTATE(O_L U,O_R V). Rotation stream root61023022 substreams1/2/3, Gaussian QR with positive diagonal signs, one triple shared. Order root61023023: permute32 parent blocks, then permute4 arms in each, saving128 fixed keys before output. This uses bounded transient all-grid factors for full physical Z GAUGE checking; no result-dependent ordering.

ALIGN uses thin QR U=QuRu,V=QvRv; SVD RuRv^T=A diag(s)B^T, Ku=A^TRu,Kv=B^TRv; output P_top Ku,Q_top Kv. It preserves both actual factor Grams, observed imbalance and prediction singular values, including pre-existing drift. Descending singular values; paired SVD signs cancel, near-tie gaps/ranks/conditions recorded. Oracle loss lower bound=.5(||C||²+sum(s²)-2 sigma_top dot s) is a construction check, not a discovered learning speedup. GAUGE Grams/imbalance are conjugated. Its physical prediction/loss/subspaces are invariant. ROTATE changes ambient directions with C fixed. All arms can have different starting loss/gradient. Matching is norm error<=1e-12+1e-10*reference norm.

Tau0..155, grid0:.02:20 plus.25 and20.2:.2:155 =1677. Full horizon; no event stopping or extension. Physical checkpoints: common0,.25,.5,1,2,5,10,20,40,80,155 plus P1 checkpoint times shifted by t*. Final exact union lists are in the input manifest. P1 comparisons use exact hundredth-tick overlap, never nearest time. Compact raw: loss, norm,32 singular values,64 signed target modes, left/right principal cosines/mean/worst/rank/condition, Gram traces/norms, nominal and observed-initial imbalance drift, loss rate. Full factors only at checkpoints; transient factors can be compared before discarding.

## Block A (before intervention integration)

For original U0=0 paths only, independently evaluate stable finite-rho graph H at81*1702=137862 points; compare raw and891 physical checkpoints. BT rank/condition/solve residual and actual C SVD vs stored basis recorded. No inverse, pseudo-inverse, jitter, seed replacement or rho=0 substitution. Independent graph error absolute<=1e-5.

Original random72 have separate posthoc mode32/33 persistence: .1 up/down brackets, last entry retained through observed end, observed duration, first interval of length10 in the piecewise-linear saved grid above threshold. Distinguish start above, end-only, insufficient remaining observation, never reached. No permanent acquisition claim. P1 first crossings remain unchanged. R8 top-block overlap and R2 cross modes at physical checkpoints only. Additional original R2 g=.025,rho=.01/1e-6 RK4 .01/.005 full0..160 =4 integrations (two halving pairs), selected after P1.

## Effects and scope

D=L_orig0-L_min fixed per target across all arms. Fixed tau0,.5,1,2,5,10,20,40,80,155:
J=Lc(0)-La(0), G=[La(0)-La(tau)]-[Lc(0)-Lc(tau)], A=Lc(tau)-La(tau)=J+G. Save raw and divided byD. G(0)=0. More favorable initial loss leaves less to reduce; negative G is not by itself intervention failure. dotL=-||gradU||²-||gradV||². Main alignment uses absolute target mean/worst errors, arm-relative half times auxiliary with ALREADY_ALIGNED/0 for ALIGN. Signed modes1/32/33 and first/re-exit/10-duration diagnostics. Fixed R paired arm differences precede descriptive groups; random n8 per cell, not128 independent datasets. No p-values or population inference. One target basis and one rotation triple, oracle information and changed starting loss/gradient are part of the intervention.

## Numerical gates, regression, costs

Inherited Zrel<=1e-6; small reference norm<=1e-8 uses Zabs<=1e-10; normalized loss<=1e-7; mean/worst<=1e-5. P1 inherited nominal imbalance relative to Gram<=1e-7; absolute nominal drift reported. New actual-initial imbalance absolute drift<=1e-8 and relative<=1e-7 registered now. Loss nonincrease normalized<=1e-9. No event-order or direction-effect gate.

P0 frozen27 and P1 frozen20 once each in separate processes/loaders. P2 meaningful tests separate. New branch refs: g=.025,R2,t*=5 all4; g=.4,R0,t*=1 CONTINUE/ALIGN, .01/.005 =>12. With block A,16 new refs; past24 excluded from new count. Full physical Z comparison for GAUGE/CONTINUE on every new grid. Original P1 all-grid stored diagnostics and exact physical checkpoint Z are compared; unsaved original full factors are not claimed restored.

Sequential BLAS1, read-only Python3.11.15/NumPy1.26.4/SciPy1.17.1/Matplotlib3.10.9. Campaign1800s, prior debit189.432583903; start remaining1610.567416097; P2 controlled worker cap900s including algebra, all tests/refs/128 integrations/summary/figures and failed attempts. Source/docs and separate raw replay measured separately. Peak RSS target2GiB, new raw/checkpoint/ref target512MiB, additional disk hard1GiB. No full P1 raw copy. Missing/retry/failure/step counts explicit, adaptive rejected steps null if unexposed. Partial on unresolved numerical/resource failure, no seed or tolerance adjustment.

Execution COMPLETE, implementation PASS and empirical findings separate. Existing P0/P1 inventory unchanged. No new GPU/model/server/install/Git write/Pages/P3; stop at P2. Finite64,finite rho,finite horizon and original-paper limiting orders are distinct.
