#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, 'sdvx_score.py')

def load_cfg(path):
    with open(path, encoding='utf-8') as f:
        cfg = json.load(f)
    if 'players' not in cfg:
        cfg['players'] = [{'fields': cfg['fields']}]
    return cfg

def missing(templates_dir, n_fields):
    out = []
    for fi in range(n_fields):
        d = os.path.join(templates_dir, f'f{fi}')
        have = set()
        if os.path.isdir(d):
            have = {os.path.splitext(x)[0] for x in os.listdir(d) if x.endswith('.png')}
        out.append({c for c in '0123456789'} - have)
    return out

def parse_pairs(text):
    pairs = []
    for chunk in text.replace('\n', ',').split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if '=' not in chunk:
            print(f'  건너뜀 (형식 오류): {chunk}', file=sys.stderr)
            continue
        t, digits = chunk.split('=', 1)
        t, digits = t.strip(), digits.strip()
        if not digits.isdigit():
            print(f'  건너뜀 (숫자 아님): {chunk}', file=sys.stderr)
            continue
        try:
            pairs.append((float(t), digits))
        except ValueError:
            print(f'  건너뜀 (시각 오류): {chunk}', file=sys.stderr)
    return pairs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--templates', default='templates')
    ap.add_argument('--pairs', default='')
    ap.add_argument('--pairs-file', default='')
    ap.add_argument('--overwrite', action='store_true',
                    help='이미 있는 숫자도 이 프레임 것으로 교체')
    args = ap.parse_args()

    text = args.pairs
    if args.pairs_file:
        with open(args.pairs_file, encoding='utf-8') as f:
            text += ',' + f.read()
    pairs = parse_pairs(text)
    if not pairs:
        raise SystemExit('학습할 "시각=점수" 항목이 없습니다.')

    cfg = load_cfg(args.config)
    n_fields = len(cfg['players'][0]['fields'])
    total_digits = sum(f['digits'] for f in cfg['players'][0]['fields'])

    print(f'{len(pairs)}개 항목으로 학습을 시작합니다 (필드 {n_fields}개, {total_digits}자리)\n')

    child_env = dict(os.environ, PYTHONIOENCODING='utf-8')
    tmpdir = tempfile.mkdtemp(prefix='batchlearn_')
    used = 0

    for t, digits in pairs:
        miss = missing(args.templates, n_fields)
        if not args.overwrite and not any(miss):
            print('\n모든 숫자를 확보해 남은 항목은 건너뜁니다.')
            break

        if len(digits) != total_digits:
            print(f'  {t:g}s  건너뜀 - {total_digits}자리여야 하는데 {len(digits)}자리 ({digits})')
            continue

        frame_path = os.path.join(tmpdir, f'f{t:g}.png')
        r = subprocess.run(
            [sys.executable, SCRIPT, 'frame', '--video', args.video,
             '--at', str(t), '--out', frame_path],
            env=child_env, capture_output=True, text=True, encoding='utf-8', errors='replace')
        if r.returncode != 0 or not os.path.exists(frame_path):
            print(f'  {t:g}s  프레임 추출 실패: {r.stderr.strip()[:80]}')
            continue

        cmd = [sys.executable, SCRIPT, 'learn', '--frame', frame_path,
               '--config', args.config, '--templates', args.templates,
               '--digits', digits]
        if args.overwrite:
            cmd.append('--overwrite')
        r = subprocess.run(cmd, env=child_env, capture_output=True, text=True,
                           encoding='utf-8', errors='replace')
        used += 1
        body = (r.stdout or r.stderr).strip().splitlines()
        summary = ' | '.join(l.strip() for l in body if '저장' in l or '보유' in l)
        print(f'  {t:g}s  {digits}  -> {summary or "변화 없음"}')

    miss = missing(args.templates, n_fields)
    print()
    for fi, m in enumerate(miss):
        state = '완료' if not m else '미보유 ' + ''.join(sorted(m))
        print(f'  field{fi}: {state}')

    if any(miss):
        print('\n아직 부족합니다. 다른 시점을 더 넣어 주세요.')
        print('(큰 숫자의 1 은 점수가 100만점을 넘어야 나옵니다 - 곡 후반을 노리세요)')
    else:
        print(f'\n모든 필드에 0~9 확보 완료 ({used}개 프레임 사용). run 으로 넘어가세요.')
        print(f'  python sdvx_score.py run --video {args.video} --config {args.config} \\')
        print(f'      --out ../videos/match.scores.json')

if __name__ == '__main__':
    main()
