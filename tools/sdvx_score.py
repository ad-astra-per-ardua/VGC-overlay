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

CELL_H = 32                                                       
SEARCH_MARGIN = 0.40                                          
MATCH_THRESHOLD = 0.45                       

def load_config(path):
    with open(path, encoding='utf-8') as f:
        cfg = json.load(f)
    cfg.setdefault('fps', 5)
    cfg.setdefault('max_score', 10_000_000)

    if 'players' not in cfg:
        cfg['players'] = [{'fields': cfg['fields']}]
    return cfg

def field_strip(frame, field):
    x, y, w, h = field['x'], field['y'], field['w'], field['h']
    roi = frame[y:y + h, x:x + w]
    if roi.size == 0:
        raise SystemExit(f'ROI 가 프레임 밖입니다: {field}')
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    hi = int(gray.max())
    bright = max(60, int(hi * 0.6))
    mask = (gray >= bright).astype(np.uint8) * 255
    rows = np.flatnonzero(mask.max(axis=1) > 0)
    if rows.size < 2:                                        
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        rows = np.flatnonzero(mask.max(axis=1) > 0)
    if rows.size >= 2:
        gray = gray[rows[0]:rows[-1] + 1, :]

    scale = CELL_H / gray.shape[0]
    strip = cv2.resize(gray, (max(1, round(gray.shape[1] * scale)), CELL_H),
                       interpolation=cv2.INTER_AREA)
    return strip

def binarize_cell(cell):
    if cell.size == 0:
        return cell
    if cell.ndim == 3:
        cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    if cell.dtype != np.uint8:
        cell = cell.astype(np.uint8)
                                    
    uniq = np.unique(cell)
    if uniq.size <= 2 and set(uniq.tolist()) <= {0, 255}:
        return cell
    _, b = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return b

def split_cells(strip, n):
    step = strip.shape[1] / n
    return [binarize_cell(strip[:, int(i * step):int((i + 1) * step)])
            for i in range(n)]

def cell_window(strip, n, i):
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

def cmd_frame(args):
    frame = read_frame_at(args.video, args.at)
    cv2.imwrite(args.out, frame)
    h, w = frame.shape[:2]
    print(f'{args.out} 저장 ({w}x{h})')
    print('그림판이나 이미지 뷰어로 열어 점수 영역의 x, y, w, h 를 확인하세요.')

def cmd_learn(args):
    cfg = load_config(args.config)
    frame = cv2.imread(args.frame)
    if frame is None:
        raise SystemExit(f'이미지를 열 수 없습니다: {args.frame}')

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
    best_digit, best_score = None, -1.0
    for d, t in templates.items():
        win = window
        if win.shape[1] < t.shape[1]:                            
            pad = t.shape[1] - win.shape[1]
            win = cv2.copyMakeBorder(win, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
        score = float(cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED).max())
        if score > best_score:
            best_digit, best_score = d, score
    return best_digit, best_score

def recognize_frame(frame, players, templates, threshold):
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
