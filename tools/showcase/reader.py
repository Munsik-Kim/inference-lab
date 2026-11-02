"""Shared reading furniture; preserves the study body and all fragment IDs."""
from html import escape
import re


def reading_layout(body, page, lang):
    if page in ('index', 'case012-annotate'):
        return body
    ko = lang == 'ko'
    def figure_link(match):
        figure = match[0]
        image = re.search(r'<img\b[^>]*\bsrc="([^"]+)"', figure)
        if not image or 'class="figure-original"' in figure:
            return figure
        return figure.replace('</figure>', '<p class="figure-original">'
            '<a href="'+escape(image[1], quote=True)+'">'
            +('그림 원본 크기로 보기' if ko else 'Open the original figure')+'</a></p></figure>')
    body = re.sub(r'<figure\b[^>]*>.*?</figure>', figure_link, body, flags=re.S)
    case = page.startswith('case')
    parent = ('전체 사례' if ko else 'All cases') if case else ('도구·안내' if ko else 'Tools and guides')
    trail = '<p class="reader-trail"><a href="index.html">DIOVA</a> / <a href="index.html#'+('cases' if case else 'tools')+'">'+parent+'</a></p>'
    # Only an existing local contents nav qualifies. Related/global navigation
    # is retained verbatim. Do not infer destinations from translated headings.
    selected = None
    for match in re.finditer(r'<nav\b[^>]*>.*?</nav>', body, re.S):
        targets = re.findall(r'href="([^"]+)"', match[0])
        if targets and all(t.startswith('#') for t in targets):
            selected = match
            break
    if selected is None:
        return trail+body
    nav = re.sub(r"</a>\s*[·|]\s*(?=<a\b)", "</a>", selected[0])
    # Retain the original contents markup/classes/IDs as the inner nav. CSS
    # scopes the new furniture without changing scientific explorer controls.
    toc = '<details class="reader-toc" open><summary>'+('이 페이지 목차' if ko else 'On this page')+'</summary>'+nav+'</details>'
    if 'class="study-layout"' in body:
        return trail+body[:selected.start()]+toc+body[selected.end():]
    clean = body[:selected.start()]+body[selected.end():]
    return trail+'<div class="reader-layout"><aside class="reader-rail">'+toc+'</aside><div class="reader-body">'+clean+'</div></div>'
