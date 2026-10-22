"""Small dependency-free HTML/SVG helpers shared by offline reports."""
import html


def table(headers, rows):
    def escape(value):
        return html.escape(str(value))
    return '<table><tr>' + ''.join('<th>' + escape(x) + '</th>' for x in headers) + '</tr>' + ''.join(
        '<tr>' + ''.join('<td>' + escape(x) + '</td>' for x in row) + '</tr>' for row in rows) + '</table>'


def chart(series, title):
    all_points = [p for _, ps in series for p in ps]
    if not all_points:
        return '<p>没有可用数据</p>'
    xmin, xmax = min(x for x, _ in all_points), max(x for x, _ in all_points)
    ymin, ymax = min(y for _, y in all_points), max(y for _, y in all_points)
    colors = ['#2563eb', '#e11d48', '#059669']
    body = ''
    for i, (name, ps) in enumerate(series):
        coords = ' '.join(f'{65+850*(x-xmin)/max(1,xmax-xmin):.2f},'
                          f'{220-180*(y-ymin)/max(1e-9,ymax-ymin):.2f}' for x,y in ps)
        body += f'<polyline points="{coords}" fill="none" stroke="{colors[i%3]}" stroke-width="2"/>'
        body += f'<text x="{65+i*270}" y="265" fill="{colors[i%3]}">{html.escape(name)}</text>'
    return (f'<h3>{html.escape(title)}</h3><svg viewBox="0 0 960 280" role="img">'
            f'<path d="M65 30V230H930" fill="none" stroke="#94a3b8"/>'
            f'<text x="5" y="45">{ymax:.4g}</text><text x="5" y="225">{ymin:.4g}</text>'
            f'<text x="65" y="247">{xmin}</text><text x="870" y="247">{xmax}</text>{body}</svg>')
