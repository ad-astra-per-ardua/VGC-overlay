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

config.json 예시 (fields 는 위에서부터 자리값이 큰 순서)
{
  "fields": [
    { "x": 1200, "y": 50, "w": 120, "h": 40, "digits": 4, "weight": 10000 },
    { "x": 1325, "y": 58, "w": 70,  "h": 28, "digits": 4, "weight": 1 }
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

    total_digits = sum(f['digits'] for f in cfg['fields'])
    digits = args.digits.strip()
    if len(digits) != total_digits:
        raise SystemExit(f'자릿수 불일치: config 는 {total_digits}자리인데 --digits 는 {len(digits)}자리')

    pos = 0
    all_missing = []
    for fi, field in enumerate(cfg['fields']):
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


def cmd_run(args):
    cfg = load_config(args.config)
    templates = [load_templates(args.templates, i) for i in range(len(cfg['fields']))]

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f'영상을 열 수 없습니다: {args.video}')

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, round(src_fps / cfg['fps']))

    samples = []          # (time, score) — 원본 인식 결과
    idx = 0
    weak = 0

    while True:
        ok = cap.grab()   # 건너뛸 프레임은 디코딩하지 않아 훨씬 빠름
        if not ok:
            break
        if idx % stride == 0:
            ok, frame = cap.retrieve()
            if ok:
                value, conf_min = 0, 1.0
                for fi, field in enumerate(cfg['fields']):
                    n = field['digits']
                    strip = field_strip(frame, field)
                    part = 0
                    for i in range(n):
                        d, conf = match_cell(cell_window(strip, n, i), templates[fi])
                        conf_min = min(conf_min, conf)
                        part = part * 10 + int(d)
                    value += part * field['weight']
                if conf_min < MATCH_THRESHOLD:
                    weak += 1
                    value = None
                samples.append((idx / src_fps, value))
        idx += 1
        if total and idx % (stride * 100) == 0:
            print(f'\r{idx}/{total}  ({idx * 100 // total}%)', end='', file=sys.stderr)

    cap.release()
    print('', file=sys.stderr)

    cleaned = postprocess(samples, cfg['max_score'])
    payload = {
        'video': os.path.basename(args.video),
        'fps': cfg['fps'],
        'points': cleaned,
    }
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, separators=(',', ':'))

    finals = [s for _, s in cleaned]
    print(f'{args.out} 저장 — 샘플 {len(cleaned)}개, 저신뢰 프레임 {weak}개')
    if finals:
        print(f'최종 점수: {finals[-1]:,}')
    print('최종 점수가 리절트 화면과 다르면 ROI 좌표를 다시 잡으세요.')


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
    r.set_defaults(func=cmd_run)

    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
