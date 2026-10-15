# Case014 P1 실행 인계

P1은 `results/p1_r2`에서30/30 핵심 조건 완료, 실패0/미실행0입니다. 보수적 worker wall358.877초, full forward3,845회입니다. CAL 선택은 다섯 표적 alpha1, block20/down_proj alpha0.75이며 비CAL 모델 평가 전에 동결했습니다.

기존 WSL snapshot을 읽기 전용으로 사용하도록 사용자 승인을 받았습니다. D 원본은 이번 실행의 선행 조건이 아닙니다. snapshot·캐시를 새 임시 파일로 오인해 정리하지 않습니다. 전체 로컬 weight hash는 기존 blob 식별자와 일치했습니다. 외부 공식 payload 인증과 구분합니다.

새 private root의 경로 mapping은 ZIP 밖 private identity/manifest에 있습니다. 입력은 기존 P0에서 동결한97개 기록을 symlink로 재사용했습니다. 새 데이터 추출·모델 다운로드·설치·P0 MC 재실행은 없습니다.

기존 P0 보고서·소스·closed manifest 전체69파일은 publication/original_pre_p1에서 원래 bytes로 보존했습니다. 기존 local_r1 결과는 수정하지 않았습니다. 현재 root 보고서/README는 실제 P1 결과를 설명합니다. 원래 PUBLIC_MANIFEST는 과거 snapshot이고 새 P1_PUBLIC_MANIFEST는 현재 검토본입니다.

독립 scalar 검산,30개 artifact 새 프로세스 독립 decode,8개 private-array Torch FP64 검산을 통과했습니다. 공개 scalar 검산은 포함되지 않은 GPU logits/weights를 재현하는 것이 아닙니다. 작은 변화의 산술 부호 불확정은 별도 사후 진단이며 원래 분류를 변경하지 않았습니다.

P2/P3, 새 selector·quantizer·모델·agent·packed serving, commit/push/PR/merge/Pages/Drive는 실행하지 않았습니다. 이 범위는 다음 실행 승인과 구분합니다.
