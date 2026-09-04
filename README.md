# SDVX 대회 방송 시스템

리듬게임(SDVX) 대회 녹화본을 방송용으로 송출하는 시스템. 영상에서 점수를 자동 인식해 타임라인을 만들고, 웹 오버레이가 재생 시각에 맞춰 점수·순위·연출을 표시한다.

---

## 구성

```
sdvx-broadcast/
├── server.js              Node 서버 (상태 관리 + SSE 브로드캐스트)
├── state.json             현재 방송 상태 (자동 저장)
├── public/
│   ├── overlay.html       OBS 브라우저 소스로 띄우는 오버레이
│   └── control.html       진행자용 컨트롤 페이지
├── presets/               라운드별 세팅 저장
├── tools/
│   ├── sdvx_score.py      점수 인식 핵심 스크립트
│   ├── batch_learn.py     템플릿 자동 학습 (보조)
│   └── make_sheet.py      시각별 점수 확인용 시트 생성
├── r1/ r3/ r4/            라운드별 설정·템플릿·점수 데이터
│   ├── config.json        ROI 좌표
│   ├── breaks.json        곡 전환 시각
│   ├── templates/         숫자 템플릿 (f0/, f1/)  ※ R4 는 templates_final/
│   ├── freeze.json        무시할 구간
│   └── match*.p*.scores.json   생성된 타임라인
└── videos/                영상 파일 (git 제외)
```

## 실행

```bash
npm start                  # http://localhost:8787
```

- 오버레이: `http://localhost:8787/overlay.html` (OBS 브라우저 소스, 1920×1080)
- 컨트롤: `http://localhost:8787/control.html` (일반 브라우저)

---

## 대회 포맷

| 라운드 | 형식 | 화면 |
|---|---|---|
| R1 | 2v2 팀 매치 | 4분할 (4320×1650) |
| R2 | 1v1 메가믹스 5곡 | 2분할 + HUD 배너 |
| R3 | 2v2 팀 매치 | 4분할 (4320×1650) |
| R4 | 1v1 싱글 | 2분할 (2160×1650) |

---

## 점수 인식

영상 → ROI 크롭 → 이진화 → 템플릿 매칭 → 후처리 → 타임라인 JSON.

자세한 구조와 함정은 **`CLAUDE.md`** 참고. 작업 전 반드시 읽을 것.

### 기본 사용

```bash
# 좌표 잡을 프레임 뽑기
python tools/sdvx_score.py frame --video "영상경로" --at 40 --out r4/f40.png

# 한 시점 인식 확인 (자릿수별 신뢰도 출력)
python tools/sdvx_score.py check --video "영상경로" --config r4/config.json \
  --player-templates-file r4/pt.json --at 40 --dump r4/dbg40

# 구간을 훑어 어디서 끊기는지 확인
python tools/sdvx_score.py scan --video "영상경로" --config r4/config.json \
  --player-templates-file r4/pt.json --start 0 --end 145 --step 10

# 전체 실행
python tools/sdvx_score.py run --video "영상경로" --config r4/config.json \
  --player-templates-file r4/pt.json --breaks-file r4/breaks.json \
  --freeze '[[146,166],[300,362],[470,490]]' --out r4/match4.scores.json
```

### 현재 설정값

| 라운드 | config / templates | breaks | freeze |
|---|---|---|---|
| R1 | `r1/config.json` / `r1/templates` | `[165, 317, 514]` | `[[647,670]]` |
| R3 | `r3/config.json` / `r3/templates` | `[562]` | `[[705,733]]` |
| R4 | `r4/config.json` / `r4/templates_final` (`r4/pt.json`) | `[162, 332, 490]` | `[[150,186],[303,351],[472,509],[608,626]]` |

breaks 는 **NEXT AREA 화면이 뜨는 시각**으로 잡는다. 리절트·중간발표·캐릭터 연출까지는 이전 곡 격차를 유지하고, 선곡 화면에서 0으로 리셋된다.

R3 타임라인은 360초부터다. 그 앞은 4분할이 아닌 3인 레이아웃이라 이 config 로 읽을 수 없다.

### 검증 기준

곡별 최종 점수가 게임 리절트 화면과 일치해야 한다. 오차가 있어도 **팀 간 격차를 뒤집지 않으면** 방송용으로 허용.

---

## 오버레이

컨트롤 페이지에서 조작하면 SSE로 오버레이에 즉시 반영된다.

**주요 기능**
- 통합 영상 1개를 크롭해 선수별 화면으로 분배 (`slots[].crop`)
- 타임라인 JSON 기반 자동 점수 미터 + 자동 순위
- 곡 선택 화면: 밴 연출, STRATEGY 카드 뒤집기
- 라운드 포맷 버튼(R1/R2/R4)이 크롭·HUD 좌표를 한 번에 세팅
- 프리셋 저장/불러오기

**미터 안정화** — 점수 격차를 지수이동평균(TAU 1.2초)으로 부드럽게 만들고, 리드 방향은 1.5초 이상 유지될 때만 바꾼다. 순간 오인식으로 미터가 튀지 않게 하기 위함.

---

## 알려진 이슈

**P1/P2 원본 해상도 차이 (해결됨)** — 선수마다 녹화 해상도가 달라(1080×1920 vs 900×1600) 편집 시 배율이 어긋나 템플릿이 안 맞던 문제. 재편집(`match_4set_new.mp4`)에서 두 소스를 같은 크기로 정규화해 **템플릿 한 세트(`r4/templates_final`)로 통일**했다. 새 영상에서도 편집 단계 정규화를 먼저 확인할 것.

**흐린 리딩 제로** — 게임 UI가 앞자리 0을 회색으로 표시. 밝은 숫자 템플릿과 매칭이 안 되므로 `0b.png` 변형 템플릿을 별도로 둔다.

**곡 전환 구간 오인식** — 페이드 인/아웃 중 노이즈가 임계값을 넘어 통과하면, 단조증가 제약 때문에 그 값이 곡 내내 유지된다. `--freeze`로 해당 구간을 통째로 무시할 것.