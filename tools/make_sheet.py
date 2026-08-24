#!/usr/bin/env python3
import argparse
import json
import os
import sys

import cv2
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

def load_cfg(path):
    with open(path, encoding='utf-8') as f:
        cfg = json.load(f)
    if 'players' not in cfg:
        cfg['players'] = [{'fields': cfg['fields']}]
    return cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--step', type=float, default=2.0, help='몇 초 간격으로 뽑을지')
    ap.add_argument('--start', type=float, default=0.0)
    ap.add_argument('--end', type=float, default=0.0, help='0 이면 영상 끝까지')
    ap.add_argument('--cols', type=int, default=4, help='가로로 몇 칸씩 배치할지')
    ap.add_argument('--scale', type=float, default=2.0, help='확대 배율 (읽기 쉽게)')
    ap.add_argument('--out', default='sheet.png')
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    fields = cfg['players'][0]['fields']                
    x0 = min(f['x'] for f in fields)
    y0 = min(f['y'] for f in fields)
    x1 = max(f['x'] + f['w'] for f in fields)
    y1 = max(f['y'] + f['h'] for f in fields)
    pad = 6
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = x1 + pad, y1 + pad

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f'영상을 열 수 없습니다: {args.video}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur = total / fps if total else 0
    end = args.end if args.end > 0 else dur
    if end <= 0:
        raise SystemExit('영상 길이를 알 수 없습니다. --end 로 직접 지정하세요.')

    times = []
    t = args.start
    while t < end:
        times.append(round(t, 2))
        t += args.step

    tiles = []
    stride_frames = max(1, int(round(args.step * fps)))
    idx = 0
    want = set(int(round(tt * fps)) for tt in times)
    grabbed = {}
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx in want:
            ok, frame = cap.retrieve()
            if ok:
                h, w = frame.shape[:2]
                crop = frame[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
                if crop.size:
                    grabbed[idx] = crop.copy()
        idx += 1
    cap.release()

    if not grabbed:
        raise SystemExit('프레임을 하나도 읽지 못했습니다. 좌표나 영상 경로를 확인하세요.')

    for tt in times:
        fi = int(round(tt * fps))
        if fi not in grabbed:
            continue
        crop = grabbed[fi]
        if args.scale != 1.0:
            crop = cv2.resize(crop, None, fx=args.scale, fy=args.scale,
                              interpolation=cv2.INTER_CUBIC)
                        
        label_h = 26
        canvas = np.zeros((crop.shape[0] + label_h, crop.shape[1], 3), np.uint8)
        canvas[label_h:] = crop
        cv2.putText(canvas, f'{tt:g}s', (6, 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(canvas)

    cols = max(1, args.cols)
    rows = (len(tiles) + cols - 1) // cols
    tw = max(t.shape[1] for t in tiles)
    th = max(t.shape[0] for t in tiles)
    gap = 8
    sheet = np.full((rows * (th + gap) + gap, cols * (tw + gap) + gap, 3), 25, np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        y = gap + r * (th + gap)
        x = gap + c * (tw + gap)
        sheet[y:y + tile.shape[0], x:x + tile.shape[1]] = tile

    cv2.imwrite(args.out, sheet)
    print(f'{args.out} 저장 - {len(tiles)}장 ({args.step}초 간격, {cols}열)')
    print('이미지를 열어 각 칸의 점수 8자리를 읽고,')
    print('batch_learn.py 에 "시각=점수" 목록으로 넘기세요.')
    print('예: python batch_learn.py --video ... --config ... --pairs "0=00775736,2=00812345"')

if __name__ == '__main__':
    main()
