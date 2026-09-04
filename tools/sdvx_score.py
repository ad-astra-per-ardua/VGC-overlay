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
MATCH_THRESHOLD = 0.35


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


def cell_window(strip, n, i, margin=SEARCH_MARGIN):
    step = strip.shape[1] / n
    lo = max(0, int(i * step - step * margin))
    hi = min(strip.shape[1], int((i + 1) * step + step * margin))
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

    if args.player < 1 or args.player > len(cfg['players']):
        raise SystemExit(f'--player 는 1~{len(cfg["players"])} 사이여야 합니다')
    fields = cfg['players'][args.player - 1]['fields']
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
    if not os.path.isdir(path):
        raise SystemExit(f'템플릿이 없습니다: {path} - learn 을 먼저 실행하세요')
    for fname in sorted(os.listdir(path)):
        if not fname.lower().endswith('.png'):
            continue
        d = fname[0]
        if d not in '0123456789':
            continue
        p = os.path.join(path, fname)
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if img.shape[0] != CELL_H:
            sc = CELL_H / img.shape[0]
            img = cv2.resize(img, (max(1, round(img.shape[1] * sc)), CELL_H),
                             interpolation=cv2.INTER_AREA)
        tpl.setdefault(d, []).append(img)
    if not tpl:
        raise SystemExit(f'템플릿이 없습니다: {path} - learn 을 먼저 실행하세요')
    missing = [d for d in '0123456789' if d not in tpl]
    if missing:
        print(f'경고: 템플릿 미보유 숫자 {"".join(missing)} - 오인식 가능', file=sys.stderr)
    return tpl


def match_cell(window, templates, offset_penalty=0.5, one_penalty=0.15):
    best_digit, best_rank_score, best_raw_score = None, -1e9, -1.0
    for d, variants in templates.items():
        for t in variants:
            win = window
            if win.shape[1] < t.shape[1]:
                pad = t.shape[1] - win.shape[1]
                win = cv2.copyMakeBorder(win, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
            result = cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED)
            _, raw_score, _, loc = cv2.minMaxLoc(result)
            x, _ = loc
            match_center = x + t.shape[1] / 2
            offset_norm = abs(match_center - win.shape[1] / 2) / win.shape[1]
            rank_score = raw_score - offset_norm * offset_penalty
            if d == '1':
                rank_score -= one_penalty
            if rank_score > best_rank_score:
                best_digit, best_rank_score, best_raw_score = d, rank_score, raw_score
    return best_digit, best_raw_score


def recognize_frame(frame, players, templates_per_player, threshold):
    out = []
    for pi, p in enumerate(players):
        templates = templates_per_player[pi]
        value, conf_min = 0, 1.0
        for fi, field in enumerate(p['fields']):
            n = field['digits']
            fixed0 = field.get('fixed_leading_zero', False)
            strip = field_strip(frame, field)
            part = 0
            for i in range(n):
                if fixed0 and i == 0:
                    part = part * 10
                    continue
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


def run_range(video, cfg, templates_per_player, start_frame, end_frame, stride, src_fps,
              progress=None, hwaccel=False, threshold=MATCH_THRESHOLD):
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
                values = recognize_frame(frame, players, templates_per_player, threshold)
                samples.append((idx / src_fps, values))
        idx += 1
        if progress and idx % (stride * 50) == 0:
            progress(idx)
    cap.release()
    return samples


def _worker(job):
    video, cfg, template_dirs, start, end, stride, src_fps, hwaccel, threshold = job
    n_fields = len(cfg['players'][0]['fields'])
    templates_per_player = load_templates_per_player(template_dirs, n_fields)
    return run_range(video, cfg, templates_per_player, start, end, stride, src_fps,
                     hwaccel=hwaccel, threshold=threshold)


def parse_timestamp(s):
    raw = s
    s = str(s).strip()
    try:
        if ':' not in s:
            return float(s)
        parts = [float(p) for p in s.split(':')]
        secs = 0.0
        for p in parts:
            secs = secs * 60 + p
        return secs
    except ValueError:
        raise SystemExit(f'--breaks 의 시각을 이해할 수 없습니다: {raw!r} '
                         f'("01:30" 또는 초 단위 숫자로 적어주세요)')


def load_breakpoints(raw, breaks_file):
    text = raw
    if breaks_file:
        with open(breaks_file, encoding='utf-8-sig') as f:
            text = f.read()
    if not text:
        return []
    text = text.lstrip('\ufeff')
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise SystemExit(f'--breaks JSON 을 읽을 수 없습니다: {e}\n'
                         f'  예: --breaks \'["01:30","03:12","04:50"]\'')
    if not isinstance(data, list):
        raise SystemExit('--breaks 는 ["01:30","03:12"] 같은 JSON 배열이어야 합니다')
    return [parse_timestamp(x) for x in data]


def load_freezes(raw, freeze_file):
    text = raw
    if freeze_file:
        with open(freeze_file, encoding='utf-8-sig') as f:
            text = f.read()
    if not text:
        return []
    text = text.lstrip('\ufeff')
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise SystemExit(f'--freeze JSON 을 읽을 수 없습니다: {e}\n'
                         f'  예: --freeze \'[["8:46","9:43"]]\'')
    if not isinstance(data, list):
        raise SystemExit('--freeze 는 [["8:46","9:43"]] 같은 [시작,끝] 쌍의 배열이어야 합니다')
    out = []
    for pair in data:
        if not isinstance(pair, list) or len(pair) != 2:
            raise SystemExit(f'--freeze 각 항목은 [시작,끝] 쌍이어야 합니다: {pair!r}')
        a, b = parse_timestamp(pair[0]), parse_timestamp(pair[1])
        if b <= a:
            raise SystemExit(f'--freeze 구간의 끝이 시작보다 뒤여야 합니다: {pair!r}')
        out.append((a, b))
    out.sort()
    return out


def resolve_template_dirs(templates_dir, player_templates_json, n_players, player_templates_file=''):
    text = player_templates_json
    if player_templates_file:
        with open(player_templates_file, encoding='utf-8-sig') as f:
            text = f.read()
    if not text:
        return [templates_dir] * n_players
    text = text.lstrip('\ufeff')
    try:
        dirs = json.loads(text)
    except json.JSONDecodeError as e:
        raise SystemExit(f'--player-templates JSON 을 읽을 수 없습니다: {e}\n'
                         f'  예: --player-templates \'["<선수1 템플릿폴더>","<선수2 템플릿폴더>"]\'')
    if not isinstance(dirs, list) or len(dirs) != n_players:
        raise SystemExit(f'--player-templates 는 선수 수({n_players})와 같은 길이의 배열이어야 합니다 '
                         f'(지금 선수 수: {n_players})')
    return dirs


def load_templates_per_player(template_dirs, n_fields):
    return [[load_templates(d, fi) for fi in range(n_fields)] for d in template_dirs]


def cmd_run(args):
    cfg = load_config(args.config)
    players = cfg['players']
    n_fields = len(players[0]['fields'])
    for p in players:
        if len(p['fields']) != n_fields:
            raise SystemExit('모든 플레이어의 fields 개수가 같아야 합니다 (템플릿 공유)')

    template_dirs = resolve_template_dirs(args.templates, args.player_templates, len(players), args.player_templates_file)
    if args.player_templates:
        print(f'선수별 템플릿 적용: {template_dirs}', file=sys.stderr)

    cap = open_video(args.video, hwaccel=args.hwaccel)
    if not cap.isOpened():
        raise SystemExit(f'영상을 열 수 없습니다: {args.video}')
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    stride = max(1, round(src_fps / cfg['fps']))

    start_sec = parse_timestamp(args.start) if args.start else 0.0
    end_sec = parse_timestamp(args.end) if args.end else None
    start_frame = max(0, round(start_sec * src_fps))
    end_frame = total_frames if end_sec is None else min(total_frames, round(end_sec * src_fps))
    if start_frame >= end_frame and total_frames > 0:
        raise SystemExit(f'--start({start_sec:g}s) 가 --end({end_sec}s) 보다 뒤입니다.')
    total = end_frame - start_frame
    if start_frame > 0 or end_frame != total_frames:
        print(f'구간 제한: {start_sec:g}s ~ {"영상 끝" if end_sec is None else f"{end_sec:g}s"} '
              f'(프레임 {start_frame}~{end_frame})', file=sys.stderr)

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
        s = start_frame
        while s < end_frame:
            e = min(end_frame, s + per)
            jobs.append((args.video, cfg, template_dirs, s, e, stride, src_fps, args.hwaccel,
                         args.threshold))
            s = e
        print(f'{len(jobs)}개 구간을 {workers}개 프로세스로 병렬 처리', file=sys.stderr)
        with mp.Pool(workers) as pool:
            chunks = pool.map(_worker, jobs)
        samples = [x for chunk in chunks for x in chunk]
        samples.sort(key=lambda x: x[0])
    else:
        templates_per_player = load_templates_per_player(template_dirs, n_fields)

        def progress(idx):
            if total:
                done = idx - start_frame
                print(f'\r{done}/{total}  ({done * 100 // total}%)', end='', file=sys.stderr)

        samples = run_range(args.video, cfg, templates_per_player, start_frame, end_frame or 1 << 62,
                            stride, src_fps, progress, hwaccel=args.hwaccel,
                            threshold=args.threshold)
        print('', file=sys.stderr)

    breakpoints = load_breakpoints(args.breaks, args.breaks_file)
    if breakpoints:
        def fmt_mmss(b):
            m, s = divmod(round(b, 2), 60)
            return f'{int(m)}:{s:05.2f}'

        pretty = ', '.join(fmt_mmss(b) for b in breakpoints)
        print(f'곡 전환 지점 {len(breakpoints)}개 적용: {pretty}', file=sys.stderr)

    freezes = load_freezes(args.freeze, args.freeze_file)
    if freezes:
        def fmt_mmss(b):
            m, s = divmod(round(b, 2), 60)
            return f'{int(m)}:{s:05.2f}'

        pretty = ', '.join(f'{fmt_mmss(a)}~{fmt_mmss(b)}' for a, b in freezes)
        print(f'무시 구간 {len(freezes)}개 적용: {pretty}', file=sys.stderr)

    multi = len(players) > 1
    for pi in range(len(players)):
        per_player = [(t, vs[pi]) for t, vs in samples]
        weak = sum(1 for _, v in per_player if v is None)
        cleaned = postprocess(per_player, cfg['max_score'], breakpoints, freezes,
                              fps=cfg['fps'], max_rate_per_sec=args.max_rate)
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

        if breakpoints:
            bounds = [0.0] + breakpoints + [float('inf')]
            for si in range(len(bounds) - 1):
                lo, hi = bounds[si], bounds[si + 1]
                seg = [s for t, s in cleaned if lo <= t < hi]
                if seg:
                    print(f'    {si + 1}번곡: 최종 {seg[-1]:,}')
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
    n_fields = len(players[0]['fields'])
    template_dirs = resolve_template_dirs(args.templates, args.player_templates, len(players), args.player_templates_file)
    templates_per_player = load_templates_per_player(template_dirs, n_fields)
    frame = read_frame_at(args.video, args.at)
    if frame is None:
        raise SystemExit(f'{args.at}초 프레임을 읽지 못했습니다')

    print(f'=== {args.at}초 ===')
    for pi, p in enumerate(players):
        templates = templates_per_player[pi]
        parts, conf_min, digits_txt = [], 1.0, []
        value = 0
        for fi, field in enumerate(p['fields']):
            n = field['digits']
            fixed0 = field.get('fixed_leading_zero', False)
            strip = field_strip(frame, field)
            part = 0
            for i in range(n):
                if fixed0 and i == 0:
                    digits_txt.append('0(고정)')
                    part = part * 10
                    continue
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
    n_fields = len(players[0]['fields'])
    template_dirs = resolve_template_dirs(args.templates, args.player_templates, len(players), args.player_templates_file)
    templates_per_player = load_templates_per_player(template_dirs, n_fields)

    cap = open_video(args.video)
    if not cap.isOpened():
        raise SystemExit(f'영상을 열 수 없습니다: {args.video}')
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    dur = total / src_fps if total else 0
    end = args.end if args.end > 0 else dur

    print(f'{args.start:g}~{end:g}초를 {args.step:g}초 간격으로 확인합니다\n')
    print(f'{"시각":>8} | ' + ' | '.join(f'{i + 1}번'.rjust(12) for i in range(len(players))))
    print('-' * (10 + 15 * len(players)))

    last_ok = [None] * len(players)
    t = args.start
    while t < end:
        frame = read_frame_at(args.video, t)
        if frame is None:
            break
        vals = recognize_frame(frame, players, templates_per_player, args.threshold)
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


def postprocess(samples, max_score, breakpoints=None, freezes=None, fps=20, max_rate_per_sec=None):
    out = []
    last = 0
    bps = sorted(breakpoints or [])
    bi = 0
    frz = sorted(freezes or [])
    fi = 0
    frame_dt = 1.0 / fps if fps else 0.05
    max_jump = (max_rate_per_sec * frame_dt) if max_rate_per_sec else None
    for t, v in samples:
        while bi < len(bps) and t >= bps[bi]:
            last = 0
            bi += 1
        while fi < len(frz) and t >= frz[fi][1]:
            fi += 1
        frozen = fi < len(frz) and frz[fi][0] <= t < frz[fi][1]
        if frozen:
            v = last
        elif v is None or v < last or v > max_score:
            v = last
        elif max_jump is not None and last > 0 and (v - last) > max_jump:
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
    l.add_argument('--player', type=int, default=1,
                   help='몇 번째 선수 좌표에서 학습할지 (1부터 시작, 기본 1). '
                        '선수마다 폰트 크기가 다른 경우 --player-templates 와 함께 씁니다.')
    l.add_argument('--overwrite', action='store_true')
    l.set_defaults(func=cmd_learn)

    r = sub.add_parser('run', help='영상 전체에서 타임라인 추출')
    r.add_argument('--video', required=True)
    r.add_argument('--config', required=True)
    r.add_argument('--templates', default='templates')
    r.add_argument('--player-templates', default='',
                   help='선수마다 폰트 크기/모양이 달라 템플릿을 따로 써야 할 때. 선수 수와 같은 '
                        '길이의 JSON 배열로 폴더 경로를 나열합니다. 예: '
                        '--player-templates \'["<선수1 템플릿폴더>","<선수2 템플릿폴더>"]\'. '
                        '지정하면 --templates 는 무시됩니다.')
    r.add_argument('--player-templates-file', default='',
                   help='--player-templates 를 파일로 줄 때. 파일 내용은 위와 같은 JSON 배열. '
                        'PowerShell 따옴표 문제를 피하려면 이 방식을 권장합니다: '
                        '\'["r4/templates_final","r4/templates_final"]\' | Out-File -Encoding utf8 r4\\pt.json')
    r.add_argument('--out', required=True)
    r.add_argument('--workers', type=int, default=0,
                   help='병렬 프로세스 수 (기본 0 = 자동. 해상도에 맞춰 메모리 한도 안에서 정합니다)')
    r.add_argument('--hwaccel', action='store_true',
                   help='GPU 디코딩 시도. 드라이버 조합에 따라 실패하며 경고만 쏟는 경우가 있어 기본은 꺼져 있습니다')
    r.add_argument('--breaks', default='',
                   help='한 영상에 곡이 여러 개 이어진 경우, 곡이 바뀌는 시각들을 JSON 배열로. '
                        '예: --breaks \'["01:30","03:12","04:50"]\' (mm:ss 또는 초 단위 숫자)')
    r.add_argument('--breaks-file', default='',
                   help='--breaks 를 파일로 줄 때. 파일 내용은 위와 같은 JSON 배열')
    r.add_argument('--freeze', default='',
                   help='이 구간 동안은 값을 무시하고 직전 값을 유지합니다(곡 종료~다음 곡 시작 '
                        '사이 무음/전환 구간용). [시작,끝] 쌍의 JSON 배열. '
                        '예: --freeze \'[["8:46","9:43"]]\'')
    r.add_argument('--freeze-file', default='',
                   help='--freeze 를 파일로 줄 때. 파일 내용은 위와 같은 JSON 배열')
    r.add_argument('--start', default='',
                   help='이 시각부터만 인식합니다(mm:ss 또는 초). 화면 소스가 중간에 바뀌는 '
                        '선수를 구간별로 따로 돌릴 때 씁니다. 기본은 영상 처음부터.')
    r.add_argument('--end', default='',
                   help='이 시각까지만 인식합니다(mm:ss 또는 초). 기본은 영상 끝까지.')
    r.add_argument('--threshold', type=float, default=MATCH_THRESHOLD,
                   help='이 값보다 신뢰도가 낮은 프레임은 인식 실패로 버립니다. '
                        f'기본 {MATCH_THRESHOLD}. 낮추면 인식률이 오르지만 오인식도 통과할 수 있고, '
                        '높이면 반대입니다.')
    r.add_argument('--max-rate', type=float, default=5_000_000,
                   help='초당 최대 점수 상승폭. 이보다 빠르게 오르면 오인식으로 보고 버립니다. '
                        '기본 5000000. 한 자리 오인식(1만점 단위)이 통과한다면 '
                        '300000~600000 정도로 낮춰보세요.')
    r.set_defaults(func=cmd_run)

    c = sub.add_parser('check', help='특정 시점의 인식 결과를 자릿수별로 확인')
    c.add_argument('--video', required=True)
    c.add_argument('--config', required=True)
    c.add_argument('--templates', default='templates')
    c.add_argument('--player-templates', default='',
                   help='run 과 동일. 선수 수와 같은 길이의 JSON 배열로 템플릿 폴더 경로 나열')
    c.add_argument('--player-templates-file', default='',
                   help='--player-templates 를 파일로 줄 때 (PowerShell 따옴표 문제 회피용)')
    c.add_argument('--at', type=float, required=True, help='확인할 시각(초)')
    c.add_argument('--threshold', type=float, default=MATCH_THRESHOLD)
    c.add_argument('--dump', default='', help='잘라낸 점수 영역을 저장할 폴더')
    c.set_defaults(func=cmd_check)

    sc = sub.add_parser('scan', help='구간을 훑어 인식이 끊기는 지점 찾기')
    sc.add_argument('--video', required=True)
    sc.add_argument('--config', required=True)
    sc.add_argument('--templates', default='templates')
    sc.add_argument('--player-templates', default='',
                    help='run 과 동일. 선수 수와 같은 길이의 JSON 배열로 템플릿 폴더 경로 나열')
    sc.add_argument('--player-templates-file', default='',
                    help='--player-templates 를 파일로 줄 때 (PowerShell 따옴표 문제 회피용)')
    sc.add_argument('--start', type=float, default=0.0)
    sc.add_argument('--end', type=float, default=0.0, help='0 이면 영상 끝까지')
    sc.add_argument('--step', type=float, default=10.0)
    sc.add_argument('--threshold', type=float, default=MATCH_THRESHOLD)
    sc.set_defaults(func=cmd_scan)

    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()