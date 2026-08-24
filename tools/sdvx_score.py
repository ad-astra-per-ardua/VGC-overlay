#!/usr/bin/env python3
"""
SDVX 녹화본에서 시간별 점수를 뽑아 JSON 타임라인으로 저장합니다.

  pip install opencv-python numpy

사용 순서
---------
1) 프레임 한 장 뽑아서 점수 영역 좌표를 확인
     python sdvx_score.py frame --video ../videos/p1.mp4 --at 30 --out frame.png

2) config.json 에 좌표를 적고, 그 프레임의 실제 점수를 알려줘 템플릿 생성
     python sdvx_score.py learn --frame frame.png --config config.json --digits 01704523

   숫자 0~9 가 전부 모일 때까지 다른 시점 프레임으로 몇 번 반복하세요.
   templates/ 에 없는 숫자가 있으면 run 단계에서 경고가 뜹니다.

3) 타임라인 추출
     python sdvx_score.py run --video ../videos/p1.mp4 --config config.json \
                              --out ../videos/p1.scores.json

   --workers 는 기본 0(자동)입니다. 해상도를 보고 메모리 한도 안에서
   프로세스 수를 정합니다 - 4320x1920 같은 통합 영상은 프레임 한 장이
   24MB 라, 코어 수만큼 띄우면 메모리 부족으로 죽습니다.
   GPU 디코딩은 --hwaccel 로 켤 수 있지만, 드라이버 조합에 따라 실패하며
   경고만 쏟고 CPU 로 폴백하는 경우가 많아 기본은 꺼져 있습니다.

config.json 예시 (fields 는 위에서부터 자리값이 큰 순서)
{
  "fields": [
    { "x": 1200, "y": 50, "w": 120, "h": 40, "digits": 4, "weight": 10000 },
    { "x": 1325, "y": 58, "w": 70,  "h": 28, "digits": 4, "weight": 1 }
  ],
  "fps": 5,
  "max_score": 10000000
}

통합 영상(4명이 한 영상)은 "fields" 대신 "players" 로 지정하면
한 번의 디코딩으로 전원을 동시에 추출합니다 (4번 따로 도는 것보다 ~3.7배 빠름).
출력은 --out 이름에 .p1 ~ .p4 가 붙습니다.
{
  "players": [
    { "fields": [ { "x": 150, ... }, { "x": 275, ... } ] },
    { "fields": [ { "x": 630, ... }, { "x": 755, ... } ] },
    ...
  ],
  "fps": 5,
  "max_score": 10000000
}
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

# 윈도우 한국어 콘솔(CP949)은 일부 유니코드 문자를 못 찍어 UnicodeEncodeError 로
# 죽습니다. 출력 스트림을 UTF-8 로 바꾸고, 그래도 안 되는 문자는 물음표로
# 흘려보내 프로그램이 중단되지 않게 합니다.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


CELL_H = 32               # 자릿수 높이를 이 값으로 정규화 -> 큰 글자/작은 글자 템플릿 공용
SEARCH_MARGIN = 0.40      # 셀 폭 대비 좌우 탐색 여유 (ROI 좌표가 어긋나도 잡아냄)
MATCH_THRESHOLD = 0.45    # 이보다 낮으면 인식 실패로 간주


# ── 공통 ────────────────────────────────────────────────────────

def load_config(path):
    with open(path, encoding='utf-8') as f:
        cfg = json.load(f)
    cfg.setdefault('fps', 5)
    cfg.setdefault('max_score', 10_000_000)
    # 하위호환: "fields" 만 있으면 1인 구성으로 취급.
    # 통합 영상(4명이 한 영상)은 "players": [{ "fields": [...] }, ...] 로 지정.
    if 'players' not in cfg:
        cfg['players'] = [{'fields': cfg['fields']}]
    return cfg


def field_strip(frame, field):
    """field ROI 를 높이 CELL_H 로 정규화한 회색조 이미지로 변환.

    폭을 고정하지 않고 높이만 맞추므로, 큰 글자와 작은 글자가 같은 템플릿을
    공유하면서도 글자 비율이 찌그러지지 않습니다.

    이진화는 여기서 하지 않고 셀별로 미룹니다. SDVX 는 앞자리 채움 0 을
    회색으로 흐리게 그리는데, 스트립 전체에 Otsu 를 한 번만 걸면 임계값이
    밝은 흰 숫자들에 맞춰져 회색 0 이 통째로 날아갑니다(0 을 1 로 오인식).
    """
    x, y, w, h = field['x'], field['y'], field['w'], field['h']
    roi = frame[y:y + h, x:x + w]
    if roi.size == 0:
        raise SystemExit(f'ROI 가 프레임 밖입니다: {field}')
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 높이 정규화 기준을 잡습니다.
    # 주의: Otsu 를 ROI 전체에 걸면 배경 밝기에 따라 회색 0 이 포함되기도,
    # 빠지기도 해서 잉크 높이가 선수마다 달라집니다(46px vs 57px). 그러면
    # 스케일이 달라져 스트립 폭이 바뀌고, 4등분 경계가 어긋나 첫 자리를
    # 통째로 놓칩니다. 그래서 밝은 흰 숫자만 기준으로 삼습니다 -
    # 흰 숫자는 어느 화면에서나 같은 높이로 그려지므로 기준이 안정적입니다.
    hi = int(gray.max())
    bright = max(60, int(hi * 0.6))
    mask = (gray >= bright).astype(np.uint8) * 255
    rows = np.flatnonzero(mask.max(axis=1) > 0)
    if rows.size < 2:      # 밝은 픽셀이 거의 없으면(페이드아웃 등) Otsu 로 대체
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        rows = np.flatnonzero(mask.max(axis=1) > 0)
    if rows.size >= 2:
        gray = gray[rows[0]:rows[-1] + 1, :]

    scale = CELL_H / gray.shape[0]
    strip = cv2.resize(gray, (max(1, round(gray.shape[1] * scale)), CELL_H),
                       interpolation=cv2.INTER_AREA)
    return strip


def binarize_cell(cell):
    """셀 하나를 이진화합니다.

    셀 안에는 숫자 하나뿐이라 Otsu 가 그 숫자의 밝기에 맞춰 임계값을 잡습니다.
    회색 0 이든 흰 8 이든 각자 기준으로 떨어져 나옵니다.
    """
    if cell.size == 0:
        return cell
    if cell.ndim == 3:
        cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    if cell.dtype != np.uint8:
        cell = cell.astype(np.uint8)
    # 이미 이진(0/255)이면 그대로 (학습된 템플릿 등)
    uniq = np.unique(cell)
    if uniq.size <= 2 and set(uniq.tolist()) <= {0, 255}:
        return cell
    _, b = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return b


def split_cells(strip, n):
    """정규화된 strip 을 자릿수만큼 균등 분할 (learn 단계 전용).

    셀별로 이진화합니다 - 회색 0 과 흰 숫자가 각자 기준으로 떨어집니다.
    """
    step = strip.shape[1] / n
    return [binarize_cell(strip[:, int(i * step):int((i + 1) * step)])
            for i in range(n)]


def cell_window(strip, n, i):
    """i 번째 자릿수 주변을 여유 있게 잘라낸 탐색 구간 (run 단계 전용).

    탐색 창 안에는 대상 숫자가 주로 들어 있으므로, 이 창 단위로 이진화하면
    회색 0 도 자기 밝기에 맞춰 살아납니다.
    """
    step = strip.shape[1] / n
    lo = max(0, int(i * step - step * SEARCH_MARGIN))
    hi = min(strip.shape[1], int((i + 1) * step + step * SEARCH_MARGIN))
    return binarize_cell(strip[:, lo:hi])


def read_frame_at(video, seconds):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f'영상을 열 수 없습니다: {video}')
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f'{seconds}s 지점에서 프레임을 읽지 못했습니다')
    return frame


# ── frame ───────────────────────────────────────────────────────

def cmd_frame(args):
    frame = read_frame_at(args.video, args.at)
    cv2.imwrite(args.out, frame)
    h, w = frame.shape[:2]
    print(f'{args.out} 저장 ({w}x{h})')
    print('그림판이나 이미지 뷰어로 열어 점수 영역의 x, y, w, h 를 확인하세요.')


# ── learn ───────────────────────────────────────────────────────

def cmd_learn(args):
    cfg = load_config(args.config)
    frame = cv2.imread(args.frame)
    if frame is None:
        raise SystemExit(f'이미지를 열 수 없습니다: {args.frame}')

    # 다인 config 여도 학습은 1번 플레이어 기준. 글자 렌더링이 전원 동일하므로
    # 한 명 것만 학습하면 템플릿을 전원이 공유합니다.
    fields = cfg['players'][0]['fields']
    total_digits = sum(f['digits'] for f in fields)
    digits = args.digits.strip()
    if len(digits) != total_digits:
        raise SystemExit(f'자릿수 불일치: config 는 {total_digits}자리인데 --digits 는 {len(digits)}자리')

    pos = 0
    all_missing = []
    for fi, field in enumerate(fields):
        n = field['digits']
        cells = split_cells(field_strip(frame, field), n)
        outdir = os.path.join(args.templates, f'f{fi}')
        os.makedirs(outdir, exist_ok=True)

        saved = []
        for ch, cell in zip(digits[pos:pos + n], cells):
            path = os.path.join(outdir, f'{ch}.png')
            if os.path.exists(path) and not args.overwrite:
                continue
            cv2.imwrite(path, cell)
            saved.append(ch)
        pos += n

        have = sorted(f[0] for f in os.listdir(outdir) if f.endswith('.png'))
        missing = [d for d in '0123456789' if d not in have]
        all_missing += missing
        print(f'  field{fi}  저장 {"".join(saved) or "-"}  보유 {"".join(have)}'
              + (f'  미보유 {"".join(missing)}' if missing else ''))

    if all_missing:
        print('미보유 숫자가 있습니다 -> 다른 시점 프레임으로 learn 을 더 돌리세요')
    else:
        print('모든 필드에 0~9 확보 완료. run 으로 넘어가세요.')


# ── run ─────────────────────────────────────────────────────────

def load_templates(path, field_index=0):
    tpl = {}
    path = os.path.join(path, f'f{field_index}')
    for d in '0123456789':
        p = os.path.join(path, f'{d}.png')
        if os.path.exists(p):
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img.shape[0] != CELL_H:
                sc = CELL_H / img.shape[0]
                img = cv2.resize(img, (max(1, round(img.shape[1] * sc)), CELL_H),
                                 interpolation=cv2.INTER_AREA)
            tpl[d] = img
    if not tpl:
        raise SystemExit(f'템플릿이 없습니다: {path} - learn 을 먼저 실행하세요')
    missing = [d for d in '0123456789' if d not in tpl]
    if missing:
        print(f'경고: 템플릿 미보유 숫자 {"".join(missing)} - 오인식 가능', file=sys.stderr)
    return tpl


def match_cell(window, templates):
    """탐색 구간 안을 템플릿으로 훑어 가장 잘 맞는 숫자와 점수를 반환."""
    best_digit, best_score = None, -1.0
    for d, t in templates.items():
        win = window
        if win.shape[1] < t.shape[1]:      # 구간이 템플릿보다 좁으면 여백을 덧댐
            pad = t.shape[1] - win.shape[1]
            win = cv2.copyMakeBorder(win, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
        score = float(cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED).max())
        if score > best_score:
            best_digit, best_score = d, score
    return best_digit, best_score


def recognize_frame(frame, players, templates, threshold):
    """한 프레임에서 모든 플레이어의 점수를 인식. [(value|None), ...] 반환."""
    out = []
    for p in players:
        value, conf_min = 0, 1.0
        for fi, field in enumerate(p['fields']):
            n = field['digits']
            strip = field_strip(frame, field)
            part = 0
            for i in range(n):
                d, conf = match_cell(cell_window(strip, n, i), templates[fi])
                conf_min = min(conf_min, conf)
                part = part * 10 + int(d)
            value += part * field['weight']
        out.append(None if conf_min < threshold else value)
    return out


def open_video(path, hwaccel=False):
    """VideoCapture 를 엽니다.

    hwaccel=True 면 GPU(NVDEC 등) 디코딩을 시도합니다. 다만 드라이버·OpenCV
    빌드 조합에 따라 d3d11va 초기화가 실패하며 경고를 쏟아내고 결국 CPU 로
    폴백하는 경우가 많아, 기본값은 꺼두고 --hwaccel 로 켜도록 했습니다.
    """
    if hwaccel:
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG,
                               [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])
        if cap.isOpened():
            return cap
    return cv2.VideoCapture(path)


def safe_workers(requested, width, height, hard_cap=None):
    if hard_cap is None:
        hard_cap = (os.cpu_count() or 4)

    per_frame_mb = (width * height * 3) / (1024 * 1024)
    if per_frame_mb <= 0:
        return max(1, min(requested, hard_cap))

    avail_mb = 2048
    try:
        import psutil
        avail_mb = psutil.virtual_memory().available / (1024 * 1024)
    except Exception:
        pass

    budget_mb = avail_mb * 0.5
    est_per_proc = per_frame_mb * 20
    fit = int(budget_mb // max(1, est_per_proc))
    return max(1, min(requested, hard_cap, fit))


def run_range(video, cfg, templates, start_frame, end_frame, stride, src_fps,
              progress=None, hwaccel=False):
    cap = open_video(video, hwaccel)
    if not cap.isOpened():
        raise SystemExit(f'영상을 열 수 없습니다: {video}')
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    players = cfg['players']
    samples = []
    idx = start_frame
    while idx < end_frame:
        ok = cap.grab()
        if not ok:
            break
        if idx % stride == 0:
            try:
                ok, frame = cap.retrieve()
            except cv2.error:
                ok, frame = False, None
            if ok and frame is not None:
                values = recognize_frame(frame, players, templates, MATCH_THRESHOLD)
                samples.append((idx / src_fps, values))
        idx += 1
        if progress and idx % (stride * 50) == 0:
            progress(idx)
    cap.release()
    return samples


def _worker(job):
    video, cfg, templates_dir, start, end, stride, src_fps, hwaccel = job
    templates = [load_templates(templates_dir, i)
                 for i in range(len(cfg['players'][0]['fields']))]
    return run_range(video, cfg, templates, start, end, stride, src_fps, hwaccel=hwaccel)


def cmd_run(args):
    cfg = load_config(args.config)
    players = cfg['players']
    n_fields = len(players[0]['fields'])
    for p in players:
        if len(p['fields']) != n_fields:
            raise SystemExit('모든 플레이어의 fields 개수가 같아야 합니다 (템플릿 공유)')

    cap = open_video(args.video, hwaccel=args.hwaccel)
    if not cap.isOpened():
        raise SystemExit(f'영상을 열 수 없습니다: {args.video}')
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    stride = max(1, round(src_fps / cfg['fps']))

    explicit = args.workers > 0
    requested = args.workers if explicit else (os.cpu_count() or 1)
    if explicit:
        workers = max(1, requested)
        advised = safe_workers(requested, vw, vh)
        if workers > advised:
            print(f'경고: 이 해상도({vw}x{vh})에서 권장치는 {advised}개입니다. '
                  f'{workers}개로 진행하니 메모리 부족이 나면 줄여 주세요.', file=sys.stderr)
    else:
        workers = safe_workers(requested, vw, vh)
        if workers < requested:
            print(f'해상도 {vw}x{vh} 라 워커를 {requested} -> {workers} 개로 정했습니다 '
                  f'(여유 메모리 기준. --workers 로 직접 지정 가능)', file=sys.stderr)
    if workers > 1 and total > 0:
        import multiprocessing as mp
        per = ((total // workers) // stride + 1) * stride
        jobs = []
        s = 0
        while s < total:
            e = min(total, s + per)
            jobs.append((args.video, cfg, args.templates, s, e, stride, src_fps, args.hwaccel))
            s = e
        print(f'{len(jobs)}개 구간을 {workers}개 프로세스로 병렬 처리', file=sys.stderr)
        with mp.Pool(workers) as pool:
            chunks = pool.map(_worker, jobs)
        samples = [x for chunk in chunks for x in chunk]
        samples.sort(key=lambda x: x[0])
    else:
        templates = [load_templates(args.templates, i) for i in range(n_fields)]
        done = [0]
        def progress(idx):
            if total:
                print(f'\r{idx}/{total}  ({idx * 100 // total}%)', end='', file=sys.stderr)
        samples = run_range(args.video, cfg, templates, 0, total or 1 << 62,
                            stride, src_fps, progress, hwaccel=args.hwaccel)
        print('', file=sys.stderr)

    multi = len(players) > 1
    for pi in range(len(players)):
        per_player = [(t, vs[pi]) for t, vs in samples]
        weak = sum(1 for _, v in per_player if v is None)
        cleaned = postprocess(per_player, cfg['max_score'])
        out_path = player_out_path(args.out, pi, multi)
        out_dir = os.path.dirname(os.path.abspath(out_path))
        if not os.path.isdir(out_dir):
            raise SystemExit(f'출력 폴더가 없습니다: {out_dir}\n'
                             f'  경로를 확인하세요. 윈도우에서는 /tmp 같은 리눅스 경로가 없습니다.')
        payload = {
            'video': os.path.basename(args.video),
            'player': pi + 1,
            'fps': cfg['fps'],
            'points': cleaned,
        }
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, separators=(',', ':'))
        finals = [s for _, s in cleaned]
        tail = f'최종 {finals[-1]:,}' if finals else '샘플 없음'
        tail_none = 0
        for _, v in reversed(per_player):
            if v is None:
                tail_none += 1
            else:
                break
        warn = ''
        if per_player and tail_none > len(per_player) * 0.1:
            secs = tail_none / max(1, cfg['fps'])
            warn = (f'  <-- 주의: 마지막 {secs:.0f}초({tail_none}샘플)가 인식 실패라 '
                    f'최종값이 그 이전에서 멈췄을 수 있습니다')
        print(f'{out_path} 저장 - 샘플 {len(cleaned)}개, 저신뢰 {weak}개, {tail}{warn}')
    print('최종 점수가 리절트 화면과 다르면 ROI 좌표를 다시 잡으세요.')


def player_out_path(out, pi, multi):
    if not multi:
        return out
    base = out
    for suffix in ('.scores.json', '.json'):
        if out.endswith(suffix):
            base = out[:-len(suffix)]
            return f'{base}.p{pi + 1}{suffix}'
    return f'{out}.p{pi + 1}'


def cmd_check(args):
    cfg = load_config(args.config)
    players = cfg['players']
    templates = [load_templates(args.templates, i)
                 for i in range(len(players[0]['fields']))]
    frame = read_frame_at(args.video, args.at)
    if frame is None:
        raise SystemExit(f'{args.at}초 프레임을 읽지 못했습니다')

    print(f'=== {args.at}초 ===')
    for pi, p in enumerate(players):
        parts, conf_min, digits_txt = [], 1.0, []
        value = 0
        for fi, field in enumerate(p['fields']):
            n = field['digits']
            strip = field_strip(frame, field)
            part = 0
            for i in range(n):
                d, conf = match_cell(cell_window(strip, n, i), templates[fi])
                conf_min = min(conf_min, conf)
                digits_txt.append(f'{d}({conf:.2f})')
                part = part * 10 + int(d)
            value += part * field['weight']
            parts.append(part)
        state = 'OK' if conf_min >= args.threshold else f'저신뢰(최저 {conf_min:.2f})'
        print(f'  {pi + 1}번: {value:,}  [{state}]')
        print(f'        자릿수: {" ".join(digits_txt)}')

    if args.dump:
        os.makedirs(args.dump, exist_ok=True)
        for pi, p in enumerate(players):
            for fi, field in enumerate(p['fields']):
                strip = field_strip(frame, field)
                path = os.path.join(args.dump, f'p{pi + 1}_f{fi}.png')
                cv2.imwrite(path, strip)
        print(f'\n잘라낸 점수 영역을 {args.dump}/ 에 저장했습니다. 눈으로 확인해 보세요.')


def cmd_scan(args):
    cfg = load_config(args.config)
    players = cfg['players']
    templates = [load_templates(args.templates, i)
                 for i in range(len(players[0]['fields']))]

    cap = open_video(args.video)
    if not cap.isOpened():
        raise SystemExit(f'영상을 열 수 없습니다: {args.video}')
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    dur = total / src_fps if total else 0
    end = args.end if args.end > 0 else dur

    print(f'{args.start:g}~{end:g}초를 {args.step:g}초 간격으로 확인합니다\n')
    print(f'{"시각":>8} | ' + ' | '.join(f'{i+1}번'.rjust(12) for i in range(len(players))))
    print('-' * (10 + 15 * len(players)))

    last_ok = [None] * len(players)
    t = args.start
    while t < end:
        frame = read_frame_at(args.video, t)
        if frame is None:
            break
        vals = recognize_frame(frame, players, templates, args.threshold)
        cells = []
        for i, v in enumerate(vals):
            if v is None:
                cells.append('  ---'.rjust(12))
            else:
                cells.append(f'{v:,}'.rjust(12))
                last_ok[i] = (t, v)
        print(f'{t:8.1f} | ' + ' | '.join(cells))
        t += args.step

    print('\n마지막으로 인식에 성공한 시점:')
    for i, ok in enumerate(last_ok):
        if ok:
            print(f'  {i + 1}번: {ok[0]:g}초에 {ok[1]:,}')
        else:
            print(f'  {i + 1}번: 한 번도 성공하지 못했습니다')
    print('\n특정 선수만 일찍 끊긴다면 그 시점을 --at 으로 check 해 보세요.')


def postprocess(samples, max_score):
    out = []
    last = 0
    for t, v in samples:
        if v is None or v < last or v > max_score:
            v = last
        last = v
        out.append([round(t, 3), v])
    return out

def main():
    p = argparse.ArgumentParser(description='SDVX 녹화본 점수 추출')
    sub = p.add_subparsers(dest='cmd', required=True)

    f = sub.add_parser('frame', help='프레임 한 장 저장 (ROI 좌표 확인용)')
    f.add_argument('--video', required=True)
    f.add_argument('--at', type=float, default=30, help='추출 시점(초)')
    f.add_argument('--out', default='frame.png')
    f.set_defaults(func=cmd_frame)

    l = sub.add_parser('learn', help='라벨링된 프레임으로 숫자 템플릿 생성')
    l.add_argument('--frame', required=True)
    l.add_argument('--config', required=True)
    l.add_argument('--digits', required=True, help='그 프레임의 실제 점수 (예: 01704523)')
    l.add_argument('--templates', default='templates')
    l.add_argument('--overwrite', action='store_true')
    l.set_defaults(func=cmd_learn)

    r = sub.add_parser('run', help='영상 전체에서 타임라인 추출')
    r.add_argument('--video', required=True)
    r.add_argument('--config', required=True)
    r.add_argument('--templates', default='templates')
    r.add_argument('--out', required=True)
    r.add_argument('--workers', type=int, default=0,
                   help='병렬 프로세스 수 (기본 0 = 자동. 해상도에 맞춰 메모리 한도 안에서 정합니다)')
    r.add_argument('--hwaccel', action='store_true',
                   help='GPU 디코딩 시도. 드라이버 조합에 따라 실패하며 경고만 쏟는 경우가 있어 기본은 꺼져 있습니다')
    r.set_defaults(func=cmd_run)

    c = sub.add_parser('check', help='특정 시점의 인식 결과를 자릿수별로 확인')
    c.add_argument('--video', required=True)
    c.add_argument('--config', required=True)
    c.add_argument('--templates', default='templates')
    c.add_argument('--at', type=float, required=True, help='확인할 시각(초)')
    c.add_argument('--threshold', type=float, default=MATCH_THRESHOLD)
    c.add_argument('--dump', default='', help='잘라낸 점수 영역을 저장할 폴더')
    c.set_defaults(func=cmd_check)

    sc = sub.add_parser('scan', help='구간을 훑어 인식이 끊기는 지점 찾기')
    sc.add_argument('--video', required=True)
    sc.add_argument('--config', required=True)
    sc.add_argument('--templates', default='templates')
    sc.add_argument('--start', type=float, default=0.0)
    sc.add_argument('--end', type=float, default=0.0, help='0 이면 영상 끝까지')
    sc.add_argument('--step', type=float, default=10.0)
    sc.add_argument('--threshold', type=float, default=MATCH_THRESHOLD)
    sc.set_defaults(func=cmd_scan)

    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
