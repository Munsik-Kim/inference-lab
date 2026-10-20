"""Standalone SVG figures regenerated only from public CSV; stdlib only."""
import argparse
import csv
import html
import json
import pathlib
import statistics


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def svg(title, labels, series, footer, unit, output, points=None):
    width, height = 960, 170 + 65 * len(labels)
    margin, plot = 290, 540
    all_values = [v for _, values, _ in series for v in values if v is not None]
    if points:
        all_values += [v for groups in points for values in groups for v in values]
    maximum = max(all_values, default=1) * 1.12 or 1
    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img"><title>{html.escape(title)}</title>',
                '<rect width="100%" height="100%" fill="#f7f8f4"/>', '<g font-family="system-ui,sans-serif" fill="#1e3a30">',
                f'<text x="24" y="32" font-size="21" font-weight="700">{html.escape(title)}</text>',
                f'<text x="24" y="58" font-size="13">{html.escape(unit)} · axis starts at0</text>']
    for i in range(5):
        x = margin + plot * i / 4
        elements.append(f'<path d="M{x} 80V{height-70}" stroke="#d9e1dc"/><text x="{x}" y="{height-50}" font-size="12">{maximum*i/4:.2f}</text>')
    for i, label in enumerate(labels):
        y = 105 + i * 65
        elements.append(f'<text x="24" y="{y+12}" font-size="14">{html.escape(label)}</text>')
        for j, (name, values, color) in enumerate(series):
            value = values[i]
            if value is None:
                continue
            yy = y + j * 20
            length = value / maximum * plot
            elements.append(f'<rect x="{margin}" y="{yy}" width="{length}" height="15" fill="{color}"/><text x="{margin+length+7}" y="{yy+12}" font-size="12">{value:.3f}</text>')
            if points:
                for k, point in enumerate(points[j][i]):
                    elements.append(f'<circle cx="{margin+point/maximum*plot}" cy="{yy+4+k*4}" r="3" fill="#172d25" stroke="white"/>')
    for j, (name, _, color) in enumerate(series):
        x = 470 + j * 160
        elements.append(f'<rect x="{x}" y="42" width="12" height="12" fill="{color}"/><text x="{x+18}" y="53" font-size="12">{html.escape(name)}</text>')
    elements.append(f'<text x="24" y="{height-14}" font-size="12">{html.escape(footer)}</text></g></svg>')
    output.write_text(''.join(elements) + '\n')


def generate(analysis, output):
    output.mkdir(parents=True, exist_ok=False)
    capacity = rows(analysis / 'capacity.csv')
    fp8 = rows(analysis / 'historical-fp8-capacity.csv')[0]
    blocks = rows(analysis / 'per-round.csv')
    fidelity = rows(analysis / 'fidelity.csv')
    # All bars are directly derived from rows; actual individual rounds remain in CSV/table.
    for lang in ('en', 'ko'):
        txt = lambda k, e: k if lang == 'ko' else e
        caps = {r['kv']: r for r in capacity}
        svg(txt('같은 KV pool의 token 수용량', 'Token slots at the same KV pool'), ['B16 · '+caps['B16']['bytes_per_token']+' B/token', 'FP8 · '+fp8['bytes_per_token']+' B/token · old P0', 'NVFP4 · '+caps['F4']['bytes_per_token']+' B/token'],
            [('Token slots', [float(caps['B16']['token_slots']), float(fp8['token_slots']), float(caps['F4']['token_slots'])], '#3e806a')],
            txt('FP8은 원 P0 로그의 실측. 최대 단일 문맥28672토큰.', 'FP8: historical P0 measurement. Maximum single context28672.'), 'token slots', output / ('capacity-' + lang + '.svg'))
        labels, rates, waits, points = [], {'B16': [], 'F4': []}, {'B16': [], 'F4': []}, {'B16': [], 'F4': []}
        for length in (8192, 16384):
            for c in (1, 2, 4):
                labels.append(f'{length//1024}K · C{c}')
                for kv in ('B16', 'F4'):
                    group = [r for r in blocks if (r['kv'], int(r['length']), int(r['concurrency'])) == (kv, length, c)]
                    rates[kv].append(statistics.median(float(r['requests_per_second']) for r in group) if group else None)
                    points[kv].append([float(r['requests_per_second']) for r in group])
                    q = [float(r['request_queue_time_seconds_mean']) for r in group if r['request_queue_time_seconds_mean']]
                    waits[kv].append(statistics.median(q) if q else None)
        svg(txt('고정8요청 block의 처리량', 'Throughput of fixed8-request blocks'), labels,
            [('B16', rates['B16'], '#3e806a'), ('NVFP4', rates['F4'], '#bd7544')],
            txt('막대:3라운드 중앙값. 점:모든 라운드. CSV·본문 표에 전체 값.', 'Bars:three-round median. Dots:every round. Full values in CSV/table.'), 'successful requests/s', output / ('throughput-' + lang + '.svg'), [points['B16'], points['F4']])
        svg(txt('엔진 요청 대기시간', 'Engine queue time'), labels,
            [('B16', waits['B16'], '#3e806a'), ('NVFP4', waits['F4'], '#bd7544')],
            txt('block별 histogram 구간 평균의3라운드 중앙값. 선점 횟수는 CSV·본문 표에 별도.', 'Three-round median of block histogram interval means. Preemptions are separate in CSV/table.'), 'seconds · lower is less waiting', output / ('queue-' + lang + '.svg'))
        exact, equal = {}, {}
        for kv in ('B16', 'F4'):
            exact[kv] = [sum(r[kv + '_exact_answer'] == 'True' for r in fidelity if int(r['length']) == length and r['kind'] == 'copy_canary') for length in (8192, 16384)]
        svg(txt('별도 copy canary 정확 일치', 'Separate copy-canary exact answers'), ['8K ·4 prompts', '16K ·4 prompts'],
            [('B16', exact['B16'], '#3e806a'), ('NVFP4', exact['F4'], '#bd7544')],
            txt('각 길이4개 문자열 복사. 일반 언어 품질이나 업무 비열등성 인증이 아님.', 'Four copied strings/length. Not general language quality or workflow noninferiority.'), 'exact answers /4', output / ('fidelity-' + lang + '.svg'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=pathlib.Path, required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    generate(args.analysis, args.output)


if __name__ == '__main__':
    main()
