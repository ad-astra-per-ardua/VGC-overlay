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

   --workers N (기본: CPU 코어 수) 으로 병렬 처리합니다. 디코딩이 전체
   시간의 90% 이상이라 코어 수에 거의 비례해 빨라집니다.

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

CELL_H = 32               # 자릿수 높이를 이 값으로 정규화 → 큰 글자/작은 글자 템플릿 공용
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
    """field ROI 를 높이 CELL_H 로 정규화한 이진 이미지로 변환.

    폭을 고정하지 않고 높이만 맞추므로, 큰 글자와 작은 글자가 같은 템플릿을
    공유하면서도 글자 비율이 찌그러지지 않습니다.
    """
    x, y, w, h = field['x'], field['y'], field['w'], field['h']
    roi = frame[y:y + h, x:x + w]
    if roi.size == 0:
        raise SystemExit(f'ROI 가 프레임 밖입니다: {field}')
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 점수 숫자는 배경보다 밝음 → Otsu 이진화로 노트·이펙트 배경을 떨궈냄
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # ROI 여백이 아니라 '글자가 실제로 차지한 높이' 를 기준으로 정규화.
    # 이렇게 해야 큰 글자 필드와 작은 글자 필드의 글자 크기가 같아집니다.
    rows = np.flatnonzero(mask.max(axis=1) > 0)
    if rows.size >= 2:
        gray = gray[rows[0]:rows[-1] + 1, :]

    scale = CELL_H / gray.shape[0]
    strip = cv2.resize(gray, (max(1, round(gray.shape[1] * scale)), CELL_H),
                       interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(strip, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary


def split_cells(strip, n):
    """정규화된 strip 을 자릿수만큼 균등 분할 (learn 단계 전용)."""
    step = strip.shape[1] / n
    return [strip[:, int(i * step):int((i + 1) * step)] for i in range(n)]


def cell_window(strip, n, i):
    """i 번째 자릿수 주변을 여유 있게 잘라낸 탐색 구간 (run 단계 전용)."""
    step = strip.shape[1] / n
    lo = max(0, int(i * step - step * SEARCH_MARGIN))
    hi = min(strip.shape[1], int((i + 1) * step + step * SEARCH_MARGIN))
    return strip[:, lo:hi]


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
        print('미보유 숫자가 있습니다 → 다른 시점 프레임으로 learn 을 더 돌리세요')
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
        raise SystemExit(f'템플릿이 없습니다: {path} — learn 을 먼저 실행하세요')
    missing = [d for d in '0123456789' if d not in tpl]
    if missing:
        print(f'경고: 템플릿 미보유 숫자 {"".join(missing)} — 오인식 가능', file=sys.stderr)
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


def open_video(path, hwaccel=True):
    """VideoCapture 를 하드웨어 디코딩 우선으로 엽니다.

    VIDEO_ACCELERATION_ANY 는 '가능하면 GPU(NVDEC 등), 안 되면 소프트웨어'
    라는 뜻이라 GPU 가 없는 환경에서도 그대로 동작합니다.
    RTX 계열은 H.264/HEVC 디코딩을 전용 유닛(NVDEC)이 처리하므로
    CPU 디코딩 대비 수 배 빠르고, --workers 병렬과도 함께 쓸 수 있습니다.
    """
    if hwaccel:
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG,
                               [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])
        if cap.isOpened():
            return cap
    return cv2.VideoCapture(path)


def run_range(video, cfg, templates, start_frame, end_frame, stride, src_fps,
              progress=None, hwaccel=True):
    """[start_frame, end_frame) 구간을 순차 디코딩하며 stride 간격으로 인식.

    디코딩이 전체 시간의 90% 이상을 차지하므로(인식은 2~3%), 이 함수를
    여러 프로세스가 서로 다른 구간에 대해 동시에 돌리는 것이 핵심 최적화입니다.
    """
    cap = open_video(video, hwaccel)
    if not cap.isOpened():
        raise SystemExit(f'영상을 열 수 없습니다: {video}')
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    players = cfg['players']
    samples = []      # (time, [v_player0, v_player1, ...])
    idx = start_frame
    while idx < end_frame:
        ok = cap.grab()   # 건너뛸 프레임은 색변환·복사를 생략해 훨씬 빠름
        if not ok:
            break
        if idx % stride == 0:
            ok, frame = cap.retrieve()
            if ok:
                values = recognize_frame(frame, players, templates, MATCH_THRESHOLD)
                samples.append((idx / src_fps, values))
        idx += 1
        if progress and idx % (stride * 50) == 0:
            progress(idx)
    cap.release()
    return samples


def _worker(job):
    """multiprocessing 워커. (video, cfg, templates_dir, start, end, stride, fps, hwaccel)"""
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

    cap = open_video(args.video, hwaccel=not args.no_hwaccel)
    if not cap.isOpened():
        raise SystemExit(f'영상을 열 수 없습니다: {args.video}')
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    stride = max(1, round(src_fps / cfg['fps']))

    workers = max(1, args.workers)
    if workers > 1 and total > 0:
        # 구간 경계를 stride 배수에 맞춰야 샘플 시각이 단일 프로세스와 동일해집니다.
        import multiprocessing as mp
        per = ((total // workers) // stride + 1) * stride
        jobs = []
        s = 0
        while s < total:
            e = min(total, s + per)
            jobs.append((args.video, cfg, args.templates, s, e, stride, src_fps, not args.no_hwaccel))
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
                            stride, src_fps, progress, hwaccel=not args.no_hwaccel)
        print('', file=sys.stderr)

    # 플레이어별로 나눠 후처리·저장
    multi = len(players) > 1
    for pi in range(len(players)):
        per_player = [(t, vs[pi]) for t, vs in samples]
        weak = sum(1 for _, v in per_player if v is None)
        cleaned = postprocess(per_player, cfg['max_score'])
        out_path = player_out_path(args.out, pi, multi)
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
        print(f'{out_path} 저장 — 샘플 {len(cleaned)}개, 저신뢰 {weak}개, {tail}')
    print('최종 점수가 리절트 화면과 다르면 ROI 좌표를 다시 잡으세요.')


def player_out_path(out, pi, multi):
    """다인 구성이면 p1.scores.json 처럼 플레이어 번호를 붙입니다."""
    if not multi:
        return out
    base = out
    for suffix in ('.scores.json', '.json'):
        if out.endswith(suffix):
            base = out[:-len(suffix)]
            return f'{base}.p{pi + 1}{suffix}'
    return f'{out}.p{pi + 1}'


def postprocess(samples, max_score):
    """SDVX 점수는 단조 증가한다는 제약으로 오인식을 걸러냅니다."""
    out = []
    last = 0
    for t, v in samples:
        if v is None or v < last or v > max_score:
            v = last          # 실패·역행·범위초과는 직전값 유지
        last = v
        out.append([round(t, 3), v])
    return out


# ── 엔트리 ──────────────────────────────────────────────────────

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
    r.add_argument('--workers', type=int, default=os.cpu_count() or 1,
                   help='병렬 프로세스 수 (기본: CPU 코어 수). 디코딩이 병목이라 코어 수에 거의 비례해 빨라집니다')
    r.add_argument('--no-hwaccel', action='store_true',
                   help='GPU(NVDEC) 디코딩 비활성화. 기본은 가능하면 GPU, 안 되면 자동으로 CPU')
    r.set_defaults(func=cmd_run)

    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
