# One Case 010, two preserved stages / 하나의 사례, 두 보존 단계

[English introduction](README.md) · [한국어 소개](README.ko.md)

The current failure-aware execution path is **v2**. The original v1 remains available for its exploratory study, original codec and failure-boundary history. Version labels identify research stages, not two projects or a claim of before/after performance improvement.

현재 실패 보존 실행 경로는 **v2**입니다. v1은 탐색 실험·원래 codec·실패 경계의 기록으로 보존합니다. 두 이름은 연구 단계이며 별도 프로젝트나 전후 성능 향상을 뜻하지 않습니다.

| Stage / 단계 | Original research path / 원래 경로 | Preserved path / 통합 경로 | Files |
| --- | --- | --- | --- |
| A / v1 | `cases/010-ckda-finite-precision-memory-horizon/` | [versions/v1/](versions/v1/) | 1,075 |
| B / v2 | `research/case010-failure-aware-v2/` | [versions/v2/](versions/v2/) | 387 |

The complete per-file mapping, sizes and hashes are in [snapshot_manifest.json](provenance/snapshot_manifest.json). The outer README/report/summary/scripts are the integration layer. Existing reports, inputs, results, code, freeze inventories and historical status inside `versions/` keep their original bytes. Full review ZIPs and their large duplicate binary patches are not nested in this source tree.

파일별 원래 경로·현재 경로·크기·hash는 위 manifest에서 확인합니다. 바깥 README·보고서·요약·scripts가 통합 계층입니다. `versions/` 안의 보고서·입력·결과·코드·freeze·당시 상태는 원래 bytes를 유지합니다. 전체 검토 ZIP과 큰 중복 binary patch는 이 source tree에 다시 넣지 않습니다.

## Original review identities / 원본 검토본 식별자

| Version | Archive | Compressed bytes | SHA256 |
| --- | --- | --- | --- |
| v1 | `diova_ckda_memory_horizon_review_20260923_v1.zip` | 30,945,886 | `5297318b3762d1ecd43ec38aed1ac21a8b902efef746108dc1f70a4d8451fa8b` |
| v2 | `diova_case010_failure_aware_v2_review_20260923.zip` | 357,220,338 | `efe3ab2586a3cec757429386a7f874b594a9893fc771c804f5871dfaa61ed9cc` |

These identify the original review archives, not a new unified archive. The scientific source base was `eb1b1e582791ad0f96603bdd2c06a96eb2d73109`; the integration's current base and publishing commit are recorded separately. The upstream pin is `ef9d108d1692387cae37f5b2d539a71826a127c1`.

이 hash는 원래 검토 archive의 값이며 새 통합 ZIP의 값이 아닙니다. 원래 source base와 현재 통합·게시 commit을 구분합니다. 역사적 `LOCAL_REVIEW`·미게시 표시도 원래 snapshot 시점의 기록입니다.

## Where to work / 실행할 위치

- Read the [v2 adapter](versions/v2/source/online_v2.py), [checkpoint contract](versions/v2/source/checkpoint.py) and [tests](versions/v2/tests/test_online_v2.py) for current behavior.
- Restore both historical trees with [restore_workspace.py](scripts/restore_workspace.py) before running original relative-path or exact-inventory checks. A new output directory is mandatory.
- Use [the model-free wrapper](scripts/check_cpu.py) for the combined audit. See [EN](REPRODUCTION.md) / [KO](REPRODUCTION.ko.md) requirements.
- Compare Stage A and B through the [unified report EN](REPORT.md) / [KO](REPORT.ko.md), keeping their sample sizes, family546/630/195 and grids separate.

현재 동작은 v2를 읽고, 원래 검사 실행은 외부 복원 공간에서 수행합니다. 보존 source의 import·상대경로·checksum을 새 위치에 맞춰 고치지 않습니다. v1 512 입력과 v2 1024 입력을 합치거나 같은 세 checkpoint를 여섯 모델로 세지 않습니다.
