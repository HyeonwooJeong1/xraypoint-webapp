# xraypoint — 하반신 정렬 자동 측정 웹앱

하반신 전장 X-ray에서 좌우 다리의 해부학적 랜드마크 15점을 자동 검출하고
임상 각도 5종(mHKA·MPTA·LDFA·JLCA·WBL)을 계산하는 웹앱입니다.

- **완전 클라이언트 사이드**: 추론이 전부 브라우저 안(WebAssembly)에서 실행됩니다.
  이미지가 서버로 전송되지 않으며, GPU·서버·설치가 필요 없습니다.
- 모델: stage1 관절 중심 검출 → 관절별 ROI 정밀 예측.
  **무릎은 EfficientNet-B0 + HRNet 앙상블**, 골반·발목은 기존 B0입니다.
  ONNX 5개와 HRNet 가중치 조각 3개, 모델 전체 약 **238MB**입니다.
- Chrome에서 검증했습니다. 다른 브라우저와 저메모리 기기의 동작은 별도 확인이 필요합니다.

## 사용법

1. 페이지 접속 → "모델 로딩 1/5…" 후 **준비 완료 — 무릎 앙상블**이 뜰 때까지 대기.
   최초 모델 다운로드 용량은 약 238MB이며 이후에는 브라우저 캐시를 사용할 수 있습니다.
2. **X-ray 열기** 버튼 또는 드래그앤드롭으로 이미지 업로드 (여러 장 가능, 순차 처리)
3. 캔버스: 휠 줌 · 드래그 이동 · 더블클릭 리셋 · **점 근처를 드래그하면 위치 수정**
   (각도 즉시 재계산, 수정점은 노란색)
4. 단축키: `P` 점 / `A` 축선 전체 / `L` 점 이름 / `1`~`5` 각도별 축선 /
   `6`·`7` 좌우 다리 / `8`·`9`·`0` 골반·무릎·발목 확대창
5. **CSV 미리보기** 후 CSV 다운로드 (각도 5종 + 15점 원본 좌표 + 수정점 목록),
   **PNG 저장**으로 점·축선·각도가 새겨진 원본 해상도 이미지 저장

좌우는 환자 기준입니다 (환자 오른쪽 다리 = 이미지 왼쪽).

## 실행 방법 (둘 중 하나)

**① 웹 링크로 접속 (권장)** — GitHub Pages 등에 올려두면 링크 클릭만으로
실행됩니다. 배포: 이 폴더 내용물을 GitHub 저장소 루트에 올리고
Settings → Pages → Branch `main` / root 선택. 별도 빌드 과정 없음.
주소는 `https://<계정>.github.io/<저장소이름>/` 형태가 됩니다.

**② 내 PC에서 로컬 실행** — `serve.bat` 더블클릭 후 브라우저에서
http://localhost:8180 접속 (자동으로 열립니다). PC에 **파이썬이 설치되어
있어야** 합니다 (없으면 python.org에서 설치, 옵션 기본값이면 됨).

주의: `index.html`을 그냥 더블클릭(file://)하면 브라우저 보안 정책 때문에
모델 로딩이 안 됩니다. 반드시 위 두 방법 중 하나로 여세요.

## 코드 구조 / 수정 가이드

| 파일 | 역할 | 이런 걸 바꾸고 싶으면 |
|---|---|---|
| `index.html` | 화면 구성·버튼·색상 | 버튼 이름/추가, 패널 배치, 색 테마(상단 `:root` CSS 변수) |
| `app.js` | 전체 로직 | 아래 상세 |
| `ensemble.js` | 같은 입력의 raw 히트맵 평균·입출력 검증·텐서 해제 | 앙상블 구성/오류 처리 변경 |
| `models/*.onnx` | 학습된 모델 5개 | 학습 저장소의 `tools/export_web_ensemble.py`로 무릎 앙상블 변환 |
| `models/knee_hrnet_cur.weights-*.bin` | HRNet FP32 외부 가중치 3조각 | ONNX와 같은 폴더에 모두 배포; 누락 시 추론 불가 |
| `models/meta.json` | 모델 입력크기·랜드마크·ROI 비율 | 모델 구성 바꿀 때 함께 갱신 (export 스크립트가 자동 생성) |
| `serve.bat` | 로컬 서버 | 포트 바꾸려면 파일 안의 `8180` 두 곳 수정 |

`app.js` 주요 위치 (상단부터):
- `ANGLE_DEFS` / `MEASURE_LINES` — 각도 정의(어느 점들을 잇는지)와 축선.
  각도 항목을 추가·수정하려면 여기.
- `calcAngles()` — 각도 계산 공식 (mHKA/MPTA/LDFA/JLCA/WBL)
- `predictLeg()` — 추론 파이프라인 (좌우 분할 → stage1 → ROI → 정밀 예측)
- `renderCanvas()` / `drawInset()` — 화면 표기 (점 크기 `arc(x, y, 2.5, …)`,
  선 굵기 `lineWidth`, 색상 hex 코드)
- `exportAnnotated()` — PNG 저장 표기 (점 `1.6 * u`, 각도 박스 위치·크기)
- `csvRows()` / `CSV_HEAD` — CSV 컬럼 구성

수정 후에는 브라우저에서 **Ctrl+F5**(강력 새로고침)로 확인하세요.

## 무릎 앙상블 (2026-11-03)

- 멤버: `knee_fix`(기존 `knee.onnx`) + `knee_hrnet_cur`.
- 동일한 grayscale 0~1, 384×384 ROI 텐서를 두 모델에 순차 입력합니다.
  11채널 **raw 히트맵을 동일 가중치로 평균한 뒤 DARK를 한 번** 적용합니다.
  개별 좌표 평균이나 다수결은 사용하지 않습니다.
- 랜드마크 순서·입력 크기·sigma·ROI 비율이 다르면 로딩을 중단합니다.
  구성원 누락/실패 시 B0 단독으로 조용히 대체하지 않습니다.
- FP32 그대로 유지했습니다. HRNet 가중치는 정밀도 변경 없이 최대 48MiB씩 나눴습니다.
  `external_data`의 파일명·경로는 ONNX 안의 location과 일치해야 합니다.
  [ONNX Runtime 외부 가중치 설명](https://onnxruntime.ai/docs/tutorials/web/large-models.html).
- 이 두 체크포인트는 과거 80/10/10 세대입니다. 공식 90/10 연구 모델이나
  새 외부400 성능 검증을 뜻하지 않습니다. 이번 검증 범위는 변환·구현의 수치 일치와 웹앱 동작입니다.

학습 저장소의 기존 Python 환경에서 `python tools/export_web_ensemble.py`로
체크포인트를 변환·검증합니다. 검증 성공 후에만 `models/meta.json`을 갱신합니다.
기존 stage1·hip·knee·ankle ONNX가 먼저 있어야 하며, 테스트용 로컬 ROI 캐시가 필요합니다.
가중치와 메타데이터를 포함한 `webapp/` 전체를 같은 배포 버전으로 반영하세요.

검증 도구(학습 저장소):
- `node --test tests/webapp_ensemble.test.cjs`: 입력 일치, 히트맵 평균, 실패·해제 회귀 검사.
- `python tools/prepare_web_ensemble_check.py` → `node tools/check_web_ensemble.cjs`:
  Playwright와 Chrome으로 실제 WASM 수치 비교, 양쪽 다리, 점 수정, CSV·PNG, 가중치 누락 검사.
  브라우저 경로는 `XRAY_BROWSER`로 지정할 수 있습니다. 외부 브라우저 요청을 차단하며,
  영상/기준값은 로컬 `runs/`에만 저장합니다. **runs/는 배포하지 않습니다.**

검증 결과 JSON은 학습 저장소의 `docs/collaboration/webapp-ensemble-*-2026-11-03.json`에 있습니다.
로컬 Chrome 154에서 무릎 ROI 한 번 약 1.7초, 전체 영상 양쪽 다리 약 7초였습니다.
기기·브라우저·첫 다운로드 상황에 따라 달라집니다.
