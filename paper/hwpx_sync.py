"""
마크다운 원고의 수정 내용을 한글 원고(hwpx)에 옮긴다.

    python paper/hwpx_sync.py --base HEAD            # git HEAD 의 md 와 지금 md 의 차이를 적용
    python paper/hwpx_sync.py --base HEAD --dry-run  # 적용할 내용만 보여 줌

`논문 최종본.hwpx` 는 `paper/본논문_재작성.md` 에서 만든 것이라 본문 문단 하나가 hwpx 의 <hp:p> 하나,
그 안의 <hp:t> 하나와 같다. 그래서 두 판의 md 를 줄(문단) 단위로 비교해

  * 바뀐 문단: 옛 문단 글을 담은 <hp:t> 를 찾아 새 글로 바꾼다.
  * 새 문단: 바로 앞 문단의 <hp:p> 를 찾아 그 뒤에 같은 서식(문단·글자 모양)으로 넣는다.
  * 표의 바뀐 칸: 칸 글을 담은 <hp:t> 를 찾아 바꾼다.

옛 글을 담은 <hp:t> 가 정확히 하나가 아니면(각주가 끼어 있어 글이 나뉜 문단 등) 그 수정은 적용하지 않고
알린다. 그런 곳과 쪽 나눔·줄 바꿈은 한글에서 직접 고친다. PDF 는 한글에서 다시 내보낸다.
"""
from __future__ import annotations
import argparse, difflib, html, os, re, subprocess, sys, zipfile

MD = "paper/본논문_재작성.md"
HWPX = "논문 최종본.hwpx"
SECTIONS = ("Contents/section0.xml", "Contents/section1.xml", "Contents/section2.xml")


def plain(line: str) -> str:
    """md 한 줄 -> hwpx 에 들어 있는 글자 (이스케이프와 강조 기호 제거)."""
    s = line.strip()
    s = re.sub(r"^#+\s*", "", s)
    s = s.replace("\\*", "*").replace("\\~", "~").replace("\\_", "_")
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    return s


def md_lines(text: str) -> list[str]:
    return [l for l in text.splitlines() if l.strip() and l.strip() != "---"]


def cells(line: str) -> list[str]:
    return [plain(c) for c in line.strip().strip("|").split("|")]


def esc(s: str) -> str:
    return html.escape(s, quote=False)


def find_t(xml: dict, text: str) -> list[tuple[str, int, int]]:
    """<hp:t>글</hp:t> 의 글이 text 와 정확히 같은 곳 (파일, 글 시작, 글 끝)."""
    hits = []
    target = esc(text)
    for name, s in xml.items():
        for m in re.finditer(r"<hp:t>([^<]*)</hp:t>", s):
            if m.group(1) == target:
                hits.append((name, m.start(1), m.end(1)))
    return hits


def para_bounds(s: str, pos: int) -> tuple[int, int]:
    """pos 를 담은 최상위 수준이 아닌, 가장 안쪽 <hp:p ...>...</hp:p> 의 범위."""
    start = s.rfind("<hp:p ", 0, pos)
    end = s.find("</hp:p>", pos) + len("</hp:p>")
    return start, end


def new_para_like(s: str, start: int, end: int, text: str) -> str:
    """start..end 문단과 같은 문단·글자 모양으로 새 문단을 만든다."""
    p = s[start:end]
    head = re.match(r"<hp:p [^>]*>", p).group(0)
    run = re.search(r"<hp:run charPrIDRef=\"(\d+)\"", p)
    cp = run.group(1) if run else "0"
    return f'{head}<hp:run charPrIDRef="{cp}"><hp:t>{esc(text)}</hp:t></hp:run></hp:p>'


def main():
    ap = argparse.ArgumentParser(description="md 원고 수정 -> hwpx")
    ap.add_argument("--base", default="HEAD", help="비교할 옛 md 의 git 리비전")
    ap.add_argument("--md", default=MD)
    ap.add_argument("--hwpx", default=HWPX)
    ap.add_argument("--out", default=None, help="저장 경로 (기본: 원본 덮어쓰기)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    old = subprocess.run(["git", "show", f"{a.base}:{a.md}"], capture_output=True, check=True).stdout.decode("utf-8")
    new = open(a.md, encoding="utf-8").read()
    ol, nl = md_lines(old), md_lines(new)

    z = zipfile.ZipFile(a.hwpx)
    infos = z.infolist()
    data = {i.filename: z.read(i.filename) for i in infos}
    xml = {n: data[n].decode("utf-8") for n in SECTIONS if n in data}

    edits, fails = [], []          # (파일, 시작, 끝, 새 글)
    sm = difflib.SequenceMatcher(a=ol, b=nl, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        olds, news = ol[i1:i2], nl[j1:j2]
        # 표: 같은 수의 행이 바뀐 경우 칸 단위로
        if olds and all(l.lstrip().startswith("|") for l in olds + news) and len(olds) == len(news):
            for lo, ln in zip(olds, news):
                for co, cn in zip(cells(lo), cells(ln)):
                    if co != cn:
                        h = find_t(xml, co)
                        if len(h) == 1:
                            edits.append((*h[0], cn, f"표 칸: {co[:40]}… -> {cn[:40]}…"))
                        else:
                            fails.append(f"표 칸({len(h)}곳): {co[:60]}")
            continue
        # 문단 바꾸기 (앞에서부터 짝지음)
        k = min(len(olds), len(news))
        for lo, ln in zip(olds[:k], news[:k]):
            h = find_t(xml, plain(lo))
            if len(h) == 1:
                edits.append((*h[0], plain(ln), f"바꿈: {plain(lo)[:50]}…"))
            else:
                fails.append(f"문단({len(h)}곳): {plain(lo)[:70]}")
        # 남는 새 문단은 직전 문단 뒤에 넣기
        if len(news) > k:
            anchor = nl[j1 + k - 1] if k > 0 else (ol[i1 - 1] if i1 > 0 else None)
            anchor_text = plain(anchor) if anchor else None
            # 직전 문단이 이번에 바뀐 문단이면 옛 글로 찾는다
            if k > 0:
                anchor_text = plain(olds[k - 1])
            h = find_t(xml, anchor_text) if anchor_text else []
            if len(h) == 1:
                name, ts, te = h[0]
                ins = "".join("\0NEWPARA\0" + plain(x) for x in news[k:])
                edits.append((name, te, te, ins, f"새 문단 {len(news) - k}개: {plain(news[k])[:50]}…"))
            else:
                fails.append(f"새 문단의 위치({len(h)}곳): {plain(news[k])[:70]}")
        if len(olds) > k:
            fails.append(f"지운 문단(직접 지울 것): {plain(olds[k])[:70]}")

    for e in edits:
        print("적용:", e[4])
    for f in fails:
        print("미적용:", f)
    if a.dry_run:
        return

    # 뒤에서부터 적용해 위치가 어긋나지 않게 한다
    for name in xml:
        s = xml[name]
        for fn, ts, te, text, _ in sorted([e for e in edits if e[0] == name], key=lambda e: -e[1]):
            if text.startswith("\0NEWPARA\0"):
                ps, pe = para_bounds(s, ts)
                paras = [t for t in text.split("\0NEWPARA\0") if t]
                s = s[:pe] + "".join(new_para_like(s, ps, pe, t) for t in paras) + s[pe:]
            else:
                s = s[:ts] + esc(text) + s[te:]
        xml[name] = s
        data[name] = s.encode("utf-8")

    out = a.out or a.hwpx
    tmp = out + ".tmp"
    with zipfile.ZipFile(tmp, "w") as zo:
        for i in infos:
            ct = zipfile.ZIP_STORED if i.filename == "mimetype" else zipfile.ZIP_DEFLATED
            zi = zipfile.ZipInfo(i.filename, date_time=i.date_time)
            zi.compress_type = ct
            zi.external_attr = i.external_attr
            zo.writestr(zi, data[i.filename])
    os.replace(tmp, out)
    print(f"저장: {out} (적용 {len(edits)}건, 미적용 {len(fails)}건)")
    if fails:
        sys.exit(1)


if __name__ == "__main__":
    main()
