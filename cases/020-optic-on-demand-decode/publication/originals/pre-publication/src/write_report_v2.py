"""Generate the bilingual local case documents from independently reaggregated CSVs."""
import pathlib,csv,json,statistics,hashlib
R=pathlib.Path(__file__).resolve().parents[1]
def read(p):return json.loads((R/p).read_text())
def csvrows(p):return list(csv.DictReader((R/p).open()))
def write(p,s):(R/p).write_text(s.strip()+'\n',encoding='utf-8')
def med(rows,key):return statistics.median(float(r[key]) for r in rows)
def span(rows,key,scale=1,digits=2):
 a=[float(r[key])/scale for r in rows];return f'{min(a):.{digits}f}–{max(a):.{digits}f}'
def main():
 lat=csvrows('results/summary/latency.csv');mem=csvrows('results/summary/memory.csv');pair=csvrows('results/summary/paired.csv');decision=read('results/summary/decision.json');freeze=read('provenance/freeze.json');env=read('provenance/environment.json');protocol=read('configs/protocol.json');inputs=read('provenance/input_manifest.json');ledger=read('results/20261026-1235/worker_ledger.json')
 sha=hashlib.sha256((R/'configs/protocol.json').read_bytes()).hexdigest();worker_sha=hashlib.sha256((R/'src/gpu_worker.py').read_bytes()).hexdigest()
 def select(rows,s,m=None,M=None,U=None):return [r for r in rows if r['shape']==s and (m is None or r['method']==m) and (M is None or int(r['M'])==M) and (U is None or int(r['U'])==U)]
 memory_table='| Shape | A resident bytes | B/C resident bytes | A / B payload ratio | NPZ bytes / arrays bytes |\n|---|---:|---:|---:|---:|\n'
 for s in ['SMALL','LARGE']:
  a=select(mem,s,'A')[0];b=select(mem,s,'B')[0];memory_table+=f"| {s} | {int(a['gpu_resident_tensor_bytes']):,} | {int(b['gpu_resident_tensor_bytes']):,} | {int(a['gpu_resident_tensor_bytes'])/int(b['gpu_resident_tensor_bytes']):.2f}× | {b['artifact_npz_bytes']} / {b['array_payload_bytes']} |\n"
 core='| Shape | M/U | Method | GPU module resident MiB | timed allocated peak MiB | timed reserved peak MiB | host median µs [block range] | host p95 µs [range] | event bracket mean/K µs [range] | setup ms [range] | gate |\n|---|---:|---|---:|---:|---:|---|---|---|---|---|\n'
 for s in ['SMALL','LARGE']:
  for M,U in [(1,1),(128,128),(128,32),(2048,2048),(2048,512)]:
   for method in ['A','B','C']:
    ls=select(lat,s,method,M,U);ms=select(mem,s,method,M,U)
    core+=f"| {s} | {M}/{U} | {method} | {int(ms[0]['gpu_resident_tensor_bytes'])/2**20:.4f} | {span(ms,'steady_allocated_peak_bytes',2**20)} | {span(ms,'steady_reserved_peak_bytes',2**20)} | {med(ls,'host_median_us'):.2f} [{span(ls,'host_median_us')}] | {med(ls,'host_p95_us'):.2f} [{span(ls,'host_p95_us')}] | {med(ls,'event_bracket_mean_per_call_us'):.2f} [{span(ls,'event_bracket_mean_per_call_us')}] | {med(ls,'setup_seconds')*1000:.2f} [{span(ls,'setup_seconds',.001)}] | PASS |\n"
 pairtable='| Shape | M/U | B/A host b0,b1,b2 | C/B host b0,b1,b2 | C−B µs b0,b1,b2 |\n|---|---:|---|---|---|\n'
 for s in ['SMALL','LARGE']:
  for M,U in [(1,1),(128,128),(128,32),(2048,2048),(2048,512)]:
   ps=sorted(select(pair,s,M=M,U=U),key=lambda r:int(r['block']));fmt=lambda key:', '.join(f'{float(r[key]):.3f}' for r in ps);pairtable+=f"| {s} | {M}/{U} | {fmt('B_over_A_host')} | {fmt('C_over_B_host')} | {fmt('C_minus_B_host_us')} |\n"
 example='| Block | A host median µs | B host median µs | C host median µs | C/B |\n|---:|---:|---:|---:|---:|\n'
 for b in range(3):
  vals={m:next(r for r in select(lat,'LARGE',m,2048,512) if int(r['block'])==b) for m in ['A','B','C']};av,bv,cv=[float(vals[m]['host_median_us']) for m in ['A','B','C']];example+=f'| {b} | {av:.2f} | {bv:.2f} | {cv:.2f} | {cv/bv:.3f} |\n'
 focus='| LARGE M2048/U512 | A | B | C |\n|---|---:|---:|---:|\n'
 for label,key in [('Timed allocated peak MiB','steady_allocated_peak_bytes'),('Timed reserved peak MiB','steady_reserved_peak_bytes')]:
  focus+='| '+label+' | '+' | '.join(span(select(mem,'LARGE',m,2048,512),key,2**20) for m in ['A','B','C'])+' |\n'
 memory_table+='\n'+focus+'\nPayload ratio and allocated/reserved/device memory are separate measurements.\n'
 identity=f"""Case020 • run `20261026-1235` • local candidate, uncommitted.

Base: `3e6f2c6b36fb2a3853cc3677bee84904d28567cf`  
Branch: `research/case020-optic-on-demand-decode-20261026-1215`  
Upstream: `{protocol['upstream_sha']}`  
Main source/input/config bundle: `{freeze['main_bundle_sha256']}`  
Protocol SHA256: `{sha}`  
GPU worker SHA256: `{worker_sha}`
"""
 commands="""```bash
# From this case directory. These commands need no model, Torch, or CUDA.
python src/audit_v2.py verify
python src/audit_v2.py summarize --out /tmp/optic-summary
MPLCONFIGDIR=/tmp/optic-mpl python src/audit_v2.py figures --out /tmp/optic-figures
python tests/test_audit_v2.py -v
```"""
 write('README.ko.md',f"""# 필요한 임베딩만 복원하면 메모리는 얼마나 줄고, 언제 느려지는가?

PLE는 **token이 각 layer에서 사용하는 추가 임베딩**이다. OPTIC의 같은 압축 표현을 전체 복원한 table에서 읽는 A, 요청마다 복원하는 B, 중복 token을 먼저 합친 뒤 복원하는 C를 비교했다. 공식 decoder를 그대로 호출하는 작은 operator 실험이다.

상주 module tensor는 SMALL **16MiB→0.143MiB**, LARGE **256MiB→0.523MiB**였다. 측정은 **30 cell /90 cell-block 모두 완료**, 모든 layer의 수치 gate를 통과했다. 실제 timed allocated peak는 module payload 외의 출력·중간 버퍼·라이브러리 잔류 할당을 포함하므로 아래 payload 배율과 다르다.

{memory_table}

중복 제거의 결과는 block별 값으로 판단한다. 전체30개 짝 중 **{decision['host_C_faster_paired_blocks']}개**에서 C가 B보다 빨랐다.3 block 모두 C가 빨랐던 조건은 **{len(decision['dedup_gain_conditions'])}개**다. 판정은 `{', '.join(decision['labels'])}`이며 아래 원점과 전체 paired 표를 함께 읽어야 한다. host/event 변동이 커서 보편적인 속도 이득으로 확대하지 않는다.

{example}

![메모리](figures/01_memory.png)

codebook은 코드 번호로 찾는 작은 벡터 모음, decoder는 그 벡터를 각 layer 임베딩으로 바꾸는 affine 식이다. resident는 호출 사이에도 남는 메모리, workspace는 호출 도중 필요한 임시 작업 공간이다. B는 code lookup→latent 구성→GEMM→bias→BF16 cast를 수행한다. C는 unique 정렬·inverse 복원과 최종 M개 출력의 비용도 낸다.

![동기화를 포함한 operator 지연시간](figures/02_latency.png)

판정: `{', '.join(decision['labels'])}`. 세 block의 방향을 함께 보는 서술적 규칙은 첫 진단 run의 변동을 본 뒤, 수정된 본 측정 전에 정했다. 통계적 사전등록을 주장하지 않으며 더 빠른 개별 block도 모두 보존한다.

CPU 재현: NumPy 2.2.6/2.3.5와 Matplotlib 3.10.8을 사용했다. [재현 명령](REPRODUCE.ko.md)과 [전체 결과·방법·실패 이력](REPORT.ko.md)을 제공한다.

{commands}

실행 범위는 같은 synthetic artifact의 raw unscaled BF16 [M,L,H] eager 조회이다. 실모델 품질·언어 능력·전체 serving의 TTFT/TPOT 결과는 측정하지 않았다. 모델/학습 데이터 다운로드, vLLM 설치, 사이트 편입, Git stage/commit/push/PR/Release를 수행하지 않았다.

공식 근거: [논문 §2·§6](https://arxiv.org/html/2610.31309v1), [고정 구현](https://github.com/VILA-Lab/optic/blob/{protocol['upstream_sha']}/src/ple_eval/compact_adapter.py). [소스 연결](SOURCE_MAP.md), [재사용/기여 경계](REUSE_MAP.md), [고지](THIRD_PARTY_NOTICES.md).

{identity}
""")
 write('README.md',f"""# How much memory does decoding only the requested embeddings save, and when is it slower?

Per-layer embeddings (PLE) are additional embeddings each token uses at each layer. This operator case compares a dense table reconstructed from the **same OPTIC artifact** (A), the unchanged official request decoder (B), and unique→official decode→inverse gather (C).

Resident module tensor storage fell from **16MiB to0.143MiB** (SMALL) and **256MiB to0.523MiB** (LARGE). All **30 cells /90 cell-blocks** completed and passed every-layer numerical gates. Timed allocated peaks include output, temporary tensors and additional retained framework allocations; payload ratios are not whole-device VRAM ratios.

{memory_table}

Across the primary grid, C was faster than B in **{decision['host_C_faster_paired_blocks']}/30 paired blocks**. **{len(decision['dedup_gain_conditions'])} conditions** had C<B in all three blocks. The decision is `{', '.join(decision['labels'])}`. Large host/event block variation prevents a general speed ranking; retain all block points and paired values.

{example}

![Memory accounting](figures/01_memory.png)

A codebook maps code indices to small vectors; the layer decoder applies an affine map. Resident storage persists between calls; workspace is temporary storage during a call. B pays for lookup, latent construction, GEMM, bias and BF16 conversion. C additionally pays for unique/sort/inverse reconstruction and returns all M rows.

![Synchronized operator latency](figures/02_latency.png)

Labels: `{', '.join(decision['labels'])}`. The descriptive three-block consistency rule was selected after observing the diagnostic run, before corrected primary timing; it is not a preregistered statistical threshold. Faster individual blocks remain visible.

[Reproduce](REPRODUCE.md) • [Full results and methods](REPORT.md) • [Korean](README.ko.md)

{commands}

The experiment uses synthetic artifacts, raw unscaled BF16 [M,L,H] output and warm-cache eager operators. It does not measure trained OPTIC quality, language capability, full-model serving, TTFT or TPOT. No model/dataset download, vLLM installation, site integration, Git stage/commit/push/PR/Release was performed.

[Paper §2/§6](https://arxiv.org/html/2610.31309v1) • [Pinned implementation](https://github.com/VILA-Lab/optic/blob/{protocol['upstream_sha']}/src/ple_eval/compact_adapter.py) • [Source map](SOURCE_MAP.md) • [Reuse](REUSE_MAP.md) • [Notices](THIRD_PARTY_NOTICES.md)

{identity}
""")
 method_en="""## Method and measurement contract

- A: the official decoder reconstructs every token into BF16 [V,L*H], in fixed256-token chunks. Compact state is deleted, garbage collection and empty_cache occur outside timing, and only dense table state remains. Setup includes loading, generation/copying into the GPU table, cleanup and synchronization. The setup coexistence peak is separate.
- B: unchanged pinned load_compact_ple/CompactOptic.forward, all layers. No bias/lookup optimization or install_compact_ple sqrt(H) scale.
- C: torch.unique(sorted=True,return_inverse=True), official decoder of U IDs, inverse index_select returning all M outputs. No precomputed unique cache.
- D: independent NumPy FP64 lookup+affine reference on stored FP16 arrays. P1 uses scalar loops versus einsum; the independent auditor uses separate per-layer matrix products. D is not timed as a competing method or claimed to be the original model embedding.

P1: V257,L3,H16,Z8,G2,D4,K256,8bit; seeds0/1/2, distinct layer weights and nonzero biases, all257tokens plus boundary/shuffled/repeated/single/multidimensional/empty inputs, zero-weight bias-only and exactly-zero controls. NPZ loading uses allow_pickle=False. Codes serialize exactly; floating outputs need not be bit-exact. There are387 arithmetic/contract-comparison records, not387 independent experiments. The official V32/L3/H16 CPU example ran first, and its stdout/stderr/exit/wall receipt is preserved.

The common input boundary validates CPU-generated int64 IDs in[0,V) before H2D. CPU negative and out-of-range IDs are explicitly rejected by both this boundary and the official index_select path. The official API silently casts int32/float inputs to long; the common boundary rejects them. Official and adapter empty outputs are BF16[0,L*H]. A meta-device mismatch probe is documented in the P1 contracts; it is outside the supported common boundary and was not timed. No illegal GPU indices were launched.

P1 gates were fixed at FP64 loop/vector rtol1e-10/atol1e-12; BF16 relativeL2≤1e-2 overall and each layer; finite required; zero-reference maxabs≤1e-6. FP64 gates are never applied to BF16-rounded values. The same numerical gates apply to all P2 cells. A/B/C happened to be bit-identical in all 90 method-pair records; bit-exact output was not a gate.

SMALL: V8192,L8,H128,Z32,G8,D4. LARGE: V32768,L16,H256,Z32,G8,D4. Dense BF16 tables are16/256MiB. These are controlled shapes, not real model shapes; Z32 differs from the paper's E4B Z644. Large storage ratios do not establish learned compression quality.

Artifact/input/order seeds61026020/61026021/61026022 use the documented SeedSequence mapping in protocol.json. M=1 has one U1 condition; M128 and2048 each have U=M and U=M/4. Repeat75 contains exactly four copies of each selected ID, then shuffles. Input manifests preserve actual U and byte hashes. Each shape uses one artifact; three process blocks are repeated executions, not three trained models or independent user workloads.

Each block shuffles30 cells. Each cell runs in a fresh serial process with autograd off. Inputs and outputs remain on GPU during timing. Validation, file IO, H2D/D2H, profiler, logging, gc and empty_cache are outside the timed functions. Output allocation is included. Warm-up20; host100 synchronized perf_counter_ns calls; event100 brackets with common K32/8/1 for M1/128/2048. Quantiles use NumPy linear. Host wall includes completion synchronization. Event elapsed/K may include launch gaps and unique-induced idle; it is not a sum of pure kernel times. Event bracket p95/K is not a single-request tail. Floors are recorded, never subtracted.

The C component diagnostics use10 additional synchronized passes per block after main timing. Dedicated correctness preflight for all inputs occurs before main timing. Their unique/decode/inverse sums need not equal C's unsplit main time. B's separate profiler trace shows lookup/GEMM/bias/cast; it was not used for latency. All seeds, inputs, tolerances and loops stayed fixed after smoke. The first-run six smoke cells executed full timing diagnostics and were already observed at the corrected source freeze. The corrected run also executed six smoke cells before main; no protocol was changed afterwards. All smoke values are excluded from primary summaries.
"""
 method_ko="""## 방법과 계측 계약

A는 공식 decoder로 전체 token을256개 고정 chunk로 복원해 GPU BF16[V,L*H] table을 만든다. setup에서 decoder와 table이 공존한 peak를 저장하고, compact 객체·임시 tensor를 제거한 뒤 측정한다. gc/empty_cache는 setup에만 있다. B는 고정 upstream forward를 그대로 호출한다. C는 매번 unique(sorted=True,return_inverse=True)→공식 decoder(U개)→inverse gather(M개)를 수행한다. install_compact_ple의 sqrt(H) 배율은 사용하지 않는다.

D는 저장된 FP16 배열을 FP64로 올려 코드 lookup과 affine 식을 계산하는 NumPy 참조다. P1은 작은 scalar loop와 einsum을 비교하고, 독립 감사는 layer별 행렬곱으로 다시 계산한다. 원래 미압축 모델의 정답이나 latency 경쟁 대상은 아니다.

P1 fixture는 V257/L3/H16/Z8/G2/D4/K256/8bit, seed0/1/2이다. 모든257개 token·전체3 layer, token0/256, 뒤섞임·반복·단일·다차원·빈 ID, weight0+bias 대조, 서로 다른 layer bias와 정확히0 출력을 검사했다. NPZ는 allow_pickle=False로 읽고 정수 코드·배열 bytes/hash를 확인했다.387개 산술/대응 기록은 독립 실험387개를 뜻하지 않는다. 먼저 공식 V32/L3/H16 예제를 실행했고 stdout/stderr/종료코드/wall을 남겼다.

공통 입력 경계는 CPU에서 만든 int64 IDs의[0,V) 범위를 확인한 뒤 측정 밖에서 GPU로 옮긴다. 음수/범위 밖은 CPU에서 공식 경로와 공통 경계가 모두 거절했다. 공식 API는 int32/float를 long으로 묵시 변환하지만 공통 경계는 거절한다. 빈 출력은 공식·adapter 모두 BF16[0,L*H]이다. meta device 불일치 probe의 관측 행동은 P1 contracts에 별도 남겼으며 지원하는 공통 입력 계약에 포함하지 않는다. GPU illegal-index assert는 실행하지 않았다.

동결 gate: FP64 loop/vector rtol1e-10·atol1e-12, BF16 overall/layer별 relativeL2≤1e-2, finite 필수,0-reference maxabs≤1e-6. BF16 반올림 값에 FP64 gate를 적용하지 않는다. 수정 본 P2는 모든 입력의 직접 A/B/C 대조를 별도 process에서 먼저 수행하고 각 cell도 같은 gate를 통과했다. A/B/C는 실제90개 method-pair 기록에서 bit-identical했으나 이를 필수 조건으로 정하지 않았다.

SMALL은 V8192/L8/H128/Z32/G8/D4, LARGE는 V32768/L16/H256/Z32/G8/D4이다. dense table은16/256MiB다. 실모델 shape가 아니며 Z32는 공식 E4B의 Z644와 다르다. payload 배율은 학습된 압축 품질을 뜻하지 않는다.

artifact/input/order seed61026020/61026021/61026022를 분리하고 protocol의 SeedSequence mapping으로 입력을 생성했다. M1은 U1 한 조건, M128/2048은 U=M 또는 U=M/4이다. repeat75는 unique ID를 정확히4회씩 넣어 섞었고 실제 U와 bytes/hash를 저장했다. shape마다 artifact는 하나다.3 block은 같은 artifact의 process 반복이며 실모델3개나 사용자 과제3개가 아니다.

block마다30 cell 순서를 고정 shuffle하고, cell마다 새 process를 순차 실행했다. autograd OFF, eager, GPU ID/출력 계약이다. 검산·I/O·H2D/D2H·profiler·로그·gc/empty_cache는 timed function 밖이다. 출력 할당은 포함한다. warm-up20회, host100회, event100 bracket이며 M1/128/2048의 K는 모든 방법에 공통32/8/1이다. p95는 NumPy linear다. host는 완료 synchronize 비용을 포함하고 event/K는 stream 경과시간이다. event에는 launch gap/unique host wait로 인한 idle이 들어갈 수 있다. bracket p95/K는 단일 요청 tail이 아니다. 계측 floor를 저장하고 빼지 않았다.

C 내부 비용은 본 계측 뒤10회 동기화 진단 pass로 측정했다. 추가 동기화가 있어 성분 합계가 원래 C 시간과 같지 않아도 된다. B의 lookup/GEMM/bias/cast trace는 별도 profiler 실행이다.첫 run의6 smoke에서도 full timing 진단 루프를 보았음을 수정 freeze에 공개했다. 수정 run도 본 측정 전6 smoke를 실행했고 이후 protocol을 바꾸지 않았다. 모든 smoke는 본 표에서 제외했다. 본 측정에 맞춰 shape·seed·허용 오차·loop를 바꾸지 않았다.
"""
 memory_en="""## Memory and environment

Inventory walks all parameters/buffers, including nonpersistent group_offsets; storage identity deduplicates aliases. Upstream compact_resident_bytes is64bytes smaller than actual B/C module storage for both shapes. NPZ file bytes, array payload, CPU metadata, common int64 IDs, BF16 output and C's unique/inverse/intermediate/final tensors are separate. Output alone costs M*L*H*2bytes. A's table is reconstructed from the compressed artifact; it is not the original uncompressed PLE.

Warm-up output references are deleted, synchronization precedes peak reset, and no gc/empty_cache is used during timing. Reserved-start retains allocator warm-up history. Setup/cold/timed allocated peaks are separate in memory.csv and raw JSON. Module storage is not the same as allocated/reserved or total VRAM. A roughly32MiB allocated residual beyond inventoried module+IDs remained after warm-up in these workers; its exact ownership was not traced, so it is not attributed to a specific kernel or counted as user-visible payload savings. CUDA context/library memory and sampled device usage are separate. NVML device samples are not continuous peaks, and WSL process attribution is limited.

Observed hardware: RTX5080(sm_120),16303MiB, driver621.01; Ryzen7 9800X3D,16 logical CPUs; Ubuntu24.04/WSL2. GPU/CPU operator runtime: Python 3.12.14, NumPy 2.3.5, Torch 2.13.0+cu130. It is a read-only reuse of a pre-existing environment, not an independent clean-environment validation. Plotting uses a separate existing Python 3.10.19/NumPy 2.2.6/Matplotlib 3.10.8 runtime. Import paths, CPU/RAM/disk/BLAS/thread settings are recorded in provenance. Child-only OMP/MKL/OpenBLAS thread counts are1; bytecode and cache destinations are task-owned. No packages were installed or upgraded.

The FP16 GEMM FP32 out_dtype probe ran on GPU, all layers/shapes/dtypes were checked, and the separate trace includes lookup/GEMM/cast. TF32 stayed false. fp16/bf16 reduced-precision reductions were disabled; their upstream-runtime defaults were true, so this policy difference is explicit. FP16 accumulation stayed false. No dtype/CPU fallback, new kernel, compile or CUDA graph was used. The2GiB gate uses allocated peak minus initialized context baseline, including setup/cold/main peaks; reserved/context/device totals are not substituted for this gate.

Before GPU execution the compute-app list was empty and device utilization0%; device memory1844MiB. After the first diagnostic run it was again empty, utilization0%, device memory1854MiB; final primary device observations are in provenance/gpu_state_after.json. No other process was stopped, reset or reconfigured. This is a scoped device observation, not an audit that all services remained unchanged.
"""
 memory_ko="""## 메모리 회계와 환경

parameter/buffer 전부를 순회하고 non-persistent group_offsets도 합산했다. alias/view는 underlying storage 기준으로 중복을 제거했다. upstream compact_resident_bytes는 두 shape에서 실제 B/C module storage보다64bytes 작다. NPZ bytes·배열 payload·CPU metadata·공통 int64 ID·BF16 출력·C unique/inverse/중간/최종 tensor를 구분한다. 출력만 M*L*H*2bytes다. A table은 같은 압축 artifact의 재구성이며 원래 미압축 PLE가 아니다.

warm-up 출력 참조를 없애고 synchronize 후 peak를 reset했다. timed loop에는 gc/empty_cache가 없다. reserved-start에는 allocator의 warm-up 이력이 남는다. setup/cold/timed allocated peak를 raw와 memory.csv에 별도 저장한다. module storage, allocated/reserved, 전체 VRAM은 같은 수치가 아니다. warm-up 뒤 module+ID inventory 외에 약32MiB의 allocated 잔류량이 있었지만 정확한 소유자를 추적하지 않았으므로 특정 kernel 비용으로 단정하지 않는다. context/library와 장치 전체 표본도 별도다. NVML 표본은 연속 최고치가 아니며 WSL의 process attribution에는 제한이 있다.

관측 장치: RTX5080(sm_120),16303MiB, driver621.01; Ryzen7 9800X3D/16 logical CPUs; Ubuntu24.04/WSL2. operator는 기존 Python 3.12.14·NumPy 2.3.5·Torch 2.13.0+cu130 환경을 읽기 전용으로 재사용했다. 독립 clean environment 검증은 아니다. 그림은 기존 Python 3.10.19·NumPy 2.2.6·Matplotlib 3.10.8에서 만들었다. import 경로·CPU/RAM/disk·BLAS/thread 설정은 provenance에 있다. OMP/MKL/OpenBLAS thread1과 cache/bytecode 설정은 자식 process에만 적용했다. 설치·업그레이드는 없었다.

GPU FP16 GEMM의 실제 FP32 out_dtype·finite·전체 layer/shape/dtype를 확인했고 별도 trace에서 lookup/GEMM/cast를 확인했다. TF32=false, FP16 accumulation=false다. fp16/bf16 reduced-precision reduction은 runtime 기본 true에서 false로 바꿔 FP32 누적 정책을 유지했다. 이 기본값과의 차이를 공개한다. 다른 dtype/CPU fallback·새 kernel·compile·CUDA graph는 없다.2GiB gate는 초기 context baseline 대비 allocated peak이며 setup/cold/main을 포함한다. reserved/context/장치 전체 사용량과 섞지 않는다.

GPU 전 compute-app 목록은 비었고 utilization0%, 장치 메모리1844MiB였다. 첫 진단 run 종료 후에도 목록은 비었고 utilization0%,1854MiB였다. 최종 본 run 종료 상태는 provenance/gpu_state_after.json에 별도로 있다. 다른 process 종료·reset·설정 변경은 하지 않았다. 이는 해당 시점의 장치 관측이며 모든 서비스의 무변경 감사는 아니다.
"""
 history_en=f"""## Execution history, preservation and limits

**Protocol correction:** run20261026-1215 performed direct P2 A/B/C comparisons after timing, although per-worker FP64 checks preceded timing. Its90 cell-blocks are retained as DIAGNOSTIC_ONLY_PROTOCOL_ORDER. Primary run20261026-1235 adds two dedicated all-condition correctness workers before smoke/main; each timed worker verifies matching source/input preflight identity. All shapes, seeds, IDs, arithmetic, tolerances, timing loops and order stayed unchanged. All prior timings were already observed and disclosed in the new freeze. The global1800s ledger includes both runs.

P0 PASS; P1 PASS; P2 COMPLETE; P3 local candidate complete. Planned30 cells/90 cell-blocks; actual30/90; failed0; missing0. GPU workers across both run IDs: **{decision['gpu_workers_total']}** (primary {decision['primary_gpu_workers']}, prior diagnostic97), total **{decision['gpu_worker_wall_seconds']:.3f}s**, **{decision['gpu_worker_budget_remaining_seconds']:.3f}s** of 1800s left. This includes worker imports/context/setup/teardown and failed-worker slots (none), including one complete protocol-order correction rerun; no budget ledger reset. All source-access/download operations were limited to7 pinned execution/doc/license files plus1 post-measurement upstream notice; no model, dataset, wheel or dependency download was required. The user-interrupted Windows→WSL switch is recorded separately; elapsed calendar time is not relabeled as GPU time.

The initial native Windows session pointed to a nonexistent C:\\home\\mskim\\Codex_main. Read-only WSL access found /home/mskim/Codex_main and the retained Git common repository /home/mskim/inference-lab-public. That checkout contains archived-file deletions; it was neither repaired nor reset. Main was fetched and verified as3e6f2c6…, public cases001–019,40 remote branches and no open PR, and no local reserved Case020 was found. A new worktree/branch was created without stage/commit/push. Preservation snapshots cover pre-existing tracked Case trees/status and four referenced file byte hashes, not every byte on the machine.

The source freeze followed observed P1 results. Probe/smoke source is preserved in provenance/source_v1 and freeze_before_smoke.json; first-main source/config/freeze also remain under source_main_v1, protocol_run_v1.json and freeze_run_v1.json. Before main timing, raw requested-output preservation and source-identity receipts were added; arithmetic, loops, tolerances and comparison paths did not change. All 6 smoke results, including their full timing loops, were already observed and excluded from main summaries. Postprocessing/report code was written after measurements and is identified by PUBLIC_MANIFEST.json, separately from the main benchmark bundle.

Initial environment-query failures (invalid Windows cwd, sandbox WSL/NVML access, Matplotlib absent from operator runtime) were resolved without installing into an existing environment. They are not numerical or GPU-worker failures. The official CPU example has its own original receipts; a second P1 run added small BF16 raw outputs while retaining the first. Figure QA initially found clipped negative C−B points; the chart limits were corrected, with the initial image retained in the owned private directory and the change recorded in provenance. No measured cell or raw sample was changed. An early auditor draft expected45 pair records; the actual design requires3 blocks×10 inputs×3 comparisons=90, corrected before delivery. An initial no-benefit draft label was corrected to unresolved variation because10 paired blocks favored C.

The independent auditor checks frozen source/input identities, planned/actual grid, raw hashes, M/U/code bytes, timing quantiles and paired ratios, all tensor payload/storage formulas, numerical gates and alternate FP64 formulas on preserved raw BF16 values. Four mutation/positive fixtures check changed timing bytes, changed U, changed dtype payload units and a valid no-Torch path. Summary files and all 6 image files were regenerated in a fresh external output under the same CPU environments. Byte equality is an observed same-environment result, not the scientific criterion or a remote CI claim. No remote CI, other machine, trained model, generation, training, serving overlay, vLLM upgrade, paid API, Drive, Pages or publication was run.

These are warm-cache operator calls for fixed repeated ID arrays. They do not represent Zipf requests, long prefill, full-model decoding, quality retention, KV-cache reallocation, or universal latency/VRAM gains. Host/event variation is real in this run; clocks/scheduler were not controlled or causally diagnosed. Three process blocks and100 internal repetitions do not justify tiny confidence intervals. All block origins and ranges are exposed.
"""
 history_ko=f"""## 이력·보존·실행 범위

**Protocol 정정:** 첫 run20261026-1215는 worker별 FP64 대조는 timing 전에 했으나 직접 P2 A/B/C 대조는 뒤에 했다.90 cell-block을 DIAGNOSTIC_ONLY_PROTOCOL_ORDER로 보존하고, 본 run20261026-1235에서 shape별 정확성 전용 process2개로10조건의 A/B/C/FP64를 먼저 확인했다. 각 timed worker는 입력/source가 맞는 preflight hash를 확인한다. shape·seed·ID·산술·허용 오차·timing loop·실행 순서는 그대로다. 이미 본 첫 run 값과 변경 이유를 새 freeze에 공개했고 두 run을 같은1800초 장부에 더했다.

P0 PASS, P1 PASS, P2 COMPLETE, P3 로컬 공개 후보 완료. 계획30 cell/90 cell-block 대비 실제30/90, 실패0·누락0이다. 두 run의 GPU worker는 총 **{decision['gpu_workers_total']}개**(수정 본 run {decision['primary_gpu_workers']}개, 이전 진단97개), 누적 **{decision['gpu_worker_wall_seconds']:.3f}초**,1800초 중 **{decision['gpu_worker_budget_remaining_seconds']:.3f}초**가 남았다. import/context/setup/종료가 포함되며 검산 순서 오류 정정으로 전체 grid를 한 번 재실행했으며 장부를 초기화하지 않았다. source는 고정된 실행/doc/license7파일과 최종 고지1파일만 받았고 모델·dataset·wheel·의존성 다운로드는 없었다. 사용자 Windows→WSL 설정 변경으로 중단된 구간을 따로 기록하며 calendar wall을 GPU 시간으로 바꾸지 않는다.

처음 Windows native의 C:\\home\\mskim\\Codex_main은 없었다. WSL에서 실제 부모와 /home/mskim/inference-lab-public 공통 Git 저장소를 찾았다. 원 checkout의 보관 정리로 생긴 삭제 상태를 복구/reset하지 않았다. main3e6f2c6…을 fetch 확인했고 공개001–019, 원격40 branch·열린 PR0·로컬 미공개 번호를 확인한 뒤 Case020 worktree/branch를 새로 만들었다. stage/commit/push는 없다. 보존 snapshot은 기존 tracked Case tree/Git 상태와 실제 참고한4파일 bytes/hash 범위이며 전 디스크 감사가 아니다.

P1 관측 뒤 source/config/input을 freeze했다. probe/smoke 원 source와 freeze는 별도로 보존했다. 본 실행 전 요청 BF16 raw 보존과 source identity receipt만 추가했고 산술·loop·허용 오차·비교 경로는 바꾸지 않았다. 이미 본6 smoke의 전체 timing 진단값은 freeze에 밝히고 본 표에서는 제외했다. 후처리·보고서 code는 측정 후 작성했으며 최종 PUBLIC_MANIFEST로 식별한다.

초기 환경 조회 실패는 Windows cwd 부재, sandbox WSL/NVML 거절, operator runtime의 Matplotlib 부재였다. 기존 환경에 설치하지 않고 해결했으며 GPU worker/수치 실패와 구분한다. 공식 CPU 예제의 원 stdout/receipt와 첫 P1은 보존했고 두 번째 P1에서 BF16 raw를 추가했다. 그림 QA에서 처음 C−B 음수 점이 잘리는 것을 발견해 축 범위를 고쳤고 초기 그림은 소유 private 경로에 보존했다. raw 표본은 바꾸지 않았다. 초기 감사 코드가 pair 수를45로 잘못 예상했으나 설계상3 block×10입력×3비교=90임을 바로잡았다.10개 짝에서 C가 빨랐으므로 초기 no-benefit 표지도 block 변동 미확정으로 고쳤다.

독립 감사는 frozen source/input·계획/실제 cell·raw hash·M/U·timing quantile/짝 비율·dtype별 payload/고유 storage·수치 gate를 재검산하고 보존 BF16에서 별도 FP64 식을 계산한다. 변조/정상 fixture4개는 timing bytes 변경·U 변경·payload 단위 변경의 거절과 Torch 없는 정상 경로를 확인한다. 새 외부 출력에서 summary와 그림6파일을 재생성해 같은 환경에서 byte 동등성을 확인했다. 이는 과학 gate나 원격 CI PASS를 대신하지 않는다. 다른 머신/원격 CI는 실행하지 않았다.

같은 ID를 반복하는 warm-cache operator 측정이다. Zipf 요청·장문 prefill·실모델 decode·언어 능력·학습된 OPTIC 품질·KV 재배분·보편적 속도/VRAM 이득은 뜻하지 않는다. host/event 변동의 원인을 clock/scheduler 하나로 단정하지 않았다.3 process block과100회 내부 반복으로 작은 신뢰구간을 만들지 않고 원점·범위를 제공한다. 모델/학습 데이터 다운로드, generation, 학습, serving overlay/vLLM 변경, 유료 API, Drive/Pages, 게시를 실행하지 않았다.
"""
 tables_en=f"""## Full core table

Cells below summarize three blocks: central latency is median of block statistics, brackets are observed min–max, memory peaks show block ranges. Host p95 is computed separately from each block's100 calls. Event column is the mean of 100 brackets/K within each block, then the median across blocks; it is not host p95 or pure kernel time. NPZ/array bytes per shape are in the resident table; all per-block values remain in CSV/raw.

{core}

## Paired ratios and absolute differences

Ratios pair the same shape/ID/block. Column order is b0,b1,b2. No averaging of unrelated blocks or combining setup with request latency.

{pairtable}

![Total and diagnostic unique cost](figures/03_unique_cost.png)
"""
 tables_ko=f"""## 전체 핵심 표

중앙 latency는3개 block 통계의 중앙값, 대괄호는 block 최솟값–최댓값이다. memory peak는 block 범위다. host p95는 각 block의100개 호출에서 계산했다. event 열은 block 안100개 bracket/K의 평균을 구한 뒤 block 중앙값으로 표시한다. host tail이나 순수 kernel 합계가 아니다. shape별 NPZ/배열 bytes는 상주량 표에 있고 모든 block 값은 CSV/raw에 남아 있다.

{core}

## 짝 비율과 절대차

같은 shape/ID/block끼리 짝지었다. b0,b1,b2 순서다. 다른 block을 합쳐 하나의 표본으로 만들거나 setup을 요청 latency에 섞지 않았다.

{pairtable}

![C−B 총차이와 별도 성분 진단](figures/03_unique_cost.png)
"""
 write('REPORT.md',f"""# Case020 — OPTIC decode execution cost

Resident module payload was reduced, while synchronized operator latency varied substantially across process blocks. All 90 primary cell-blocks passed numerical/resource gates. C was faster than B in{decision['host_C_faster_paired_blocks']}/30 paired blocks;{len(decision['dedup_gain_conditions'])} conditions favored C in all three. Decision: `{', '.join(decision['labels'])}`.

{memory_table}

{example}

{memory_en}

{method_en}

{tables_en}

{history_en}

[Protocol](configs/protocol.json) • [Raw run](results/20261026-1235) • [Decision](results/summary/decision.json) • [Source map](SOURCE_MAP.md) • [Reproduce](REPRODUCE.md)

{identity}
""")
 write('REPORT.ko.md',f"""# Case020 — OPTIC 압축 PLE 실행 비용

상주 module payload는 줄었고 동기화를 포함한 operator latency는 process block 사이에서 크게 변했다.90개 본 cell-block 모두 수치·자원 gate를 통과했다.30개 짝 중{decision['host_C_faster_paired_blocks']}개에서 C가 B보다 빨랐고 세 block 모두 C가 빨랐던 조건은{len(decision['dedup_gain_conditions'])}개다. 판정: `{', '.join(decision['labels'])}`.

{memory_table}

{example}

{memory_ko}

{method_ko}

{tables_ko}

{history_ko}

[Protocol](configs/protocol.json) • [Raw](results/20261026-1235) • [판정](results/summary/decision.json) • [Source map](SOURCE_MAP.md) • [재현](REPRODUCE.ko.md)

{identity}
""")
 reproduce_en=f"""# Reproduce Case020

## CPU audit and reconstruction of artifacts

Use Python with NumPy 2.2.6 or2.3.5; plotting additionally needs Matplotlib 3.10.8. The CPU audit does not import Torch, initialize CUDA or need a model. --help is also CPU-only. The observed rendering runtime was Python 3.10.19; the official operator example requires Python≥3.12.

{commands}

Pipeline: `results/20261026-1235/*.json` plus `inputs/` and preserved BF16 words → `src/audit_v2.py` → `results/summary/{{correctness,memory,latency,paired}}.csv` and `decision.json` → `src/figures.py` → `figures/01_memory,02_latency,03_unique_cost.(png,svg)`. A/B/C timing samples are never pooled across blocks. The independent auditor recomputes statistics and storage formulas rather than trusting stored PASS/summary fields.

The raw B block0 NPZ files hold requested BF16 uint16 words, not dense tables. Decode words with `(words.astype(uint32)<<16).view(float32)`. P1 raw outputs cover all methods/seeds/controls. NPZ loading always uses allow_pickle=False.

## Re-run the official CPU example / P1

These commands require the tested operator runtime or another Python≥3.12 runtime with NumPy and compatible Torch. They do not run generation/download entry points. Write into a new output location to retain original evidence.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=vendor/optic/src python vendor/optic/examples/decode_optic.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B src/cpu_checks.py --out /tmp/optic-p1
```

P1 rewrites deterministic `inputs/p1_*.npz` and fixture_manifest.json; run in a disposable copy if preserving the candidate exactly. The minimal original source is vendored with hashes/license and needs no editable install or vLLM/Transformers.

## GPU re-execution protocol

The original run is complete and must not be overwritten or appended as another replicate. For a new GPU run, make a separate worktree/output copy, assign a new run ID and ledger, and update/freeze paths explicitly before running prepare→freeze→probe→preflight→smoke→sweep. The supplied scripts intentionally bind the original run ID and1800s ledger; their current commands are for audit/review, not a one-command reset of the original evidence. The worker supports individual --out locations for source review, but launching it alone bypasses the parent budget ledger and is not the recommended measured workflow.

Use a free GPU, Python≥3.12 and sm_120-supporting Torch with actual FP16 mm(out_dtype=FP32) support. Keep2GiB allocated-delta and1800s cumulative worker caps. Never replace the official arithmetic with a fallback. No package upgrade into a serving environment is needed. No weights/datasets are part of this case.

Observed operator executable: `/home/mskim/inference-lab/.venv-baseline/bin/python`; rendering executable: `/home/mskim/miniforge3/envs/ai_lab/bin/python`. Both were read-only reused. No clean install, remote CI or other-machine result is claimed.

## Verify delivered bytes

`PUBLIC_MANIFEST.json` records every case file except itself, by relative path/bytes/SHA256. It separates Git base, upstream source and uncommitted benchmark bundle. `provenance/reproduction_receipt.json` records same-environment summary/image regeneration and visual QA. No wheel/cache/model/dense table is public.
"""
 write('REPRODUCE.md',reproduce_en)
 write('REPRODUCE.ko.md',f"""# Case020 재현

## 모델·GPU 없이 원자료 재검산

NumPy 2.2.6 또는2.3.5가 있는 Python을 사용한다. 그림에는 Matplotlib 3.10.8도 필요하다. verify/summarize/figures와 --help는 Torch를 import하거나 CUDA context를 만들지 않는다. 실제 그림 runtime은 Python 3.10.19이며 공식 operator 예제는 Python≥3.12가 필요하다.

{commands}

경로: `results/20261026-1235/*.json`+`inputs/`+보존 BF16 raw → `src/audit_v2.py` → `results/summary/{{correctness,memory,latency,paired}}.csv`·`decision.json` → `src/figures.py` → `figures/01_memory,02_latency,03_unique_cost.(png,svg)`. 저장된 PASS/summary를 정답으로 쓰지 않고 raw timing·storage 식·수치 gate를 재계산한다. 다른 block 표본은 합치지 않는다.

B block0의10개 NPZ는 dense table이 아닌 요청 출력의 BF16 uint16 words다. `(words.astype(uint32)<<16).view(float32)`로 해석한다. P1 raw에는 모든 method/seed/control의257개 출력이 있다. NPZ는 allow_pickle=False로 읽는다.

## 공식 CPU 예제와 P1

테스트한 operator runtime 또는 NumPy/Torch가 있는 Python≥3.12에서 실행한다. generation/자동 다운로드 entry point를 실행하지 않는다.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=vendor/optic/src python vendor/optic/examples/decode_optic.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B src/cpu_checks.py --out /tmp/optic-p1
```

P1은 deterministic inputs/p1_*.npz와 fixture_manifest.json을 다시 쓰므로 전달 후보의 bytes를 보존하려면 별도 사본에서 실행한다. vendored 최소 source/라이선스/hash를 사용하며 editable install·vLLM·Transformers가 필요하지 않다.

## GPU를 새로 측정할 때

현재 run은 완료됐으며 덮어쓰거나 다른 반복으로 덧붙이지 않는다. 새 worktree/output 사본에 새 run ID/장부를 지정하고 경로를 명시적으로 바꿔 freeze한 뒤 prepare→freeze→probe→preflight→smoke→sweep 순서로 수행한다. 제공된 orchestration은 원 run ID와1800초 장부에 고정되어 있다. 원자료를 reset하는 단축 명령을 제공하지 않는다. 개별 worker만 띄우면 상위 장부를 우회하므로 권장 계측 경로가 아니다.

GPU compute가 비어 있어야 하며 Python≥3.12·sm_120 지원 Torch·실제 FP16 mm(out_dtype=FP32) probe 통과가 필요하다. 추가 allocated2GiB·누적 worker1800초를 유지한다. dtype/CPU fallback으로 공식 경로 PASS를 만들지 않는다. serving 환경 설치/업그레이드나 모델·dataset 다운로드는 필요하지 않다.

로컬 관측 operator executable은 `/home/mskim/inference-lab/.venv-baseline/bin/python`, 그림 executable은 `/home/mskim/miniforge3/envs/ai_lab/bin/python`이다. 둘 다 읽기 전용 재사용이며 clean install/다른 머신/원격 CI 검증을 뜻하지 않는다.

## 파일 확인

PUBLIC_MANIFEST.json은 자기 자신을 제외한 Case 파일의 상대경로·bytes·SHA256을 담고 Git base/upstream/미커밋 benchmark bundle을 구분한다. provenance/reproduction_receipt.json에 같은 환경에서 summary/그림 재생성 및 시각 QA를 남겼다. wheel/cache/model/dense table 사본은 공개 산출물에 넣지 않았다.
""")
 write('SOURCE_MAP.md',f"""# Source map — pinned OPTIC to this operator case

| Source | Verified meaning | Local use / boundary |
|---|---|---|
| [Paper v1 §2.2/§2.3](https://arxiv.org/html/2610.31309v1) | Token-shared discrete codes, layer-specific affine decoders; output-path fitting | Context and affine form only; no fitting or model-quality reproduction |
| [Paper §6](https://arxiv.org/html/2610.31309v1) | Compact inference background | Paper RTX4090/full-model measurements are not Case020 numbers |
| [compact_adapter.py@{protocol['upstream_sha']}](https://github.com/VILA-Lab/optic/blob/{protocol['upstream_sha']}/src/ple_eval/compact_adapter.py) | load_compact_ple selects optic_npz; discrete,8bits,K256; Z=G*D; exact geometry | Vendored unchanged. uint8 token codes; FP16 codebooks/weights/bias; metadata and bytes in source/fixture manifests |
| CompactOptic.forward | Flatten IDs→code index_select→group offsets/embedding→latent→stacked GEMM→FP32 bias→BF16 [*ids.shape,L*H] | B calls original forward. All layers. No built-in torch.unique; C is an explicit wrapper |
| _matmul_fp32 | CUDA torch.mm(FP16,FP16,out_dtype=FP32); CPU casts operands toFP32 | Actual sm_120 probe, no fallback. CPU and GPU arithmetic paths differ |
| compact_resident_bytes | Counts persistent tensors, excludes nonpersistent buffers | Preserved comparison column; main accounting adds group_offsets and deduplicates underlying storage |
| install_compact_ple | Wraps decoder output with sqrt(H) scaling | Not called. A/B/C use raw unscaled load_compact_ple output |
| [CPU example](https://github.com/VILA-Lab/optic/blob/{protocol['upstream_sha']}/examples/decode_optic.py) | V32,L3,H16,G2,D4,Z8 synthetic; BF16 CPU output | Executed first; original stdout/receipt preserved. P1 extends toV257/nonzero bias/three seeds |
| [deployment.md](https://github.com/VILA-Lab/optic/blob/{protocol['upstream_sha']}/docs/deployment.md) | Python3.12,24GiB full-model example; vLLM0.19.1 serving | Background only; not required for this operator and not substituted for RTX5080 results |
| pyproject.toml / src/ple_eval/__init__.py | Python≥3.12, empty project dependencies, minimal package init | No project install; module import from pinned vendor path; NumPy/Torch suffice |
| LICENSE | Apache-2.0 | Original bytes/hash retained; notices below |

Independent reference axes: codes[V,G]; codebooks[G,K,D]; z[m,g*D+d]=codebooks[g,codes[ids[m],g],d]; weight[L,H,Z]; bias[L,H]; y[m,l,h]=sum_z z[m,z]*weight[l,h,z]+bias[l,h]. Stored FP16 values are promoted toFP64 before the sum. Layers remain in source order. The scalar loop, einsum and independent per-layer matrix products use the same documented axes but do not call the Torch decoder for reference arithmetic.

Original source identity: `{protocol['upstream_sha']}`. Exact URLs/bytes/SHA256 for 8 files: [source_manifest.json](provenance/source_manifest.json). No floating upstream main, package init workaround, .git copy, model or unused benchmark dataset was vendored.
""")
 write('REUSE_MAP.md',"""# Reuse and contribution boundary

| Component | Owner / reuse | Case020 contribution |
|---|---|---|
| OPTIC representation/official forward | VILA-Lab/optic, Apache-2.0, pinned commit | Read, freeze, execute without modification |
| GEMM, unique, gather, allocator, profiler | PyTorch/CUDA and their upstream implementations | Operator invocation/policy selection/accounting; no new kernel, codec or allocator |
| Fixture/metadata and ID streams | Case020 | Small synthetic NPZs, nonzero layer biases, exact duplication and hash manifests |
| A/C wrappers | Case020 using official B decoder | Full reconstruction/gather and unique/decode/inverse with matched outputs |
| NumPy FP64 references | Case020 | Scalar loop, vector check, separate auditor formula over stored FP16 arrays |
| Measurement/audit/figures/docs | Case020, with Codex assistance | Isolated process budget ledger, storage inventories, raw timing, independent CPU regeneration and bilingual explanation |

The root inference-lab Apache-2.0 license applies to new case code. Upstream notices remain separate. Source/protocol/input identity is distinct from Git base; no commit, publication, remote CI or clean-environment validation is implied by this candidate.
""")
 write('THIRD_PARTY_NOTICES.md',f"""# Third-party notices

OPTIC source, package metadata, official CPU example and deployment documentation are copied unchanged from [VILA-Lab/optic](https://github.com/VILA-Lab/optic/tree/{protocol['upstream_sha']}) at commit `{protocol['upstream_sha']}`. The complete upstream Apache License 2.0 text is retained at [vendor/optic/LICENSE](vendor/optic/LICENSE). Source filenames, original URLs, bytes and SHA256 are in provenance/source_manifest.json. The original Apache-2.0 LICENSE and [upstream THIRD_PARTY_NOTICES.md](vendor/optic/THIRD_PARTY_NOTICES.md) are retained unchanged. The upstream notice was added during the final license completeness check, after measurement; this did not change measured source. Its references to serving overlays, datasets and checkpoints concern upstream material not copied or executed here. Relative links inside preserved upstream documents refer to the original repository context. No additional upstream copyright assertion is introduced.

Paper: Jiacheng Liu et al., *Extreme Compression of Per-Layer Embeddings via Cross-Layer Latent Vector Quantization*, arXiv:2610.31309v1,22October2026. [Original](https://arxiv.org/abs/2610.31309), [HTML](https://arxiv.org/html/2610.31309v1). The paper's OPTIC idea and model-quality/full-model performance results belong to its authors. Case020 does not relicense the paper or bundle datasets/models.

Runtime libraries (NumPy, PyTorch, CUDA and Matplotlib) were reused from existing local installations. Their binaries, wheels, caches and transitive dependencies are not redistributed in this case. New comparison/fixture/audit/report code is provided under the inference-lab root Apache-2.0 license. Codex assisted implementation, verification and writing; this does not assign authorship of OPTIC, GEMM or the allocator to Case020.
""")
 write('requirements-cpu.txt','numpy==2.2.6\nmatplotlib==3.10.8')
 print('wrote bilingual documents and maps')
if __name__=='__main__':main()
