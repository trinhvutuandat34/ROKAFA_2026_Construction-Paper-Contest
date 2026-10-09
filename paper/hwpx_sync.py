"""
마크다운 원고의 수정 내용을 한글 원고(hwpx)에 옮긴다.

    python paper/hwpx_sync.py --base HEAD            # git HEAD 의 md 와 지금 md 의 차이를 적용
    python paper/hwpx_sync.py --base HEAD --dry-run  # 적용할 내용만 보여 줌
    python paper/hwpx_sync.py --base HEAD --image image2=results/paper_figs_rewrite/fig4_1_plane.png

`논문 최종본.hwpx` 는 `paper/본논문_재작성.md` 에서 만든 것이라 본문 문단 하나가 hwpx 의 <hp:p> 하나,
그 안의 <hp:t> 하나와 같다(각주가 있는 문단은 각주 앞뒤로 <hp:t> 가 나뉜다). 두 판의 md 를 줄(문단) 단위로 비교해

  * 바뀐 문단: 옛 글을 담은 <hp:t> 를 찾아 새 글로 바꾼다. 각주 때문에 글이 나뉜 문단은 바뀐 부분과
    그 앞뒤 몇 글자를 담은 <hp:t> 를 찾아 그 부분만 바꾼다.
  * 새 문단: 바로 앞 문단 뒤에 넣는다. 제목(#)·표 제목(<표 …>)·註·出處·본문은 원고에 이미 있는 같은
    종류의 문단에서 문단·글자 모양을 가져오고, 새 제목 앞에는 원고처럼 빈 문단을 하나 둔다.
  * 새 표: 열 수가 같은 원고의 표를 본떠 새로 만든다. 기존 표에 더한 행은 그 표에 행으로 넣는다.
  * 표의 바뀐 칸: 칸 글을 담은 <hp:t> 를 찾아 바꾼다.
  * --image: BinData 의 그림 파일을 바꾸고 그림의 원본 크기 정보를 새 그림에 맞춘다.

찾지 못한 수정은 적용하지 않고 알린다. 쪽 나눔·표 열 너비·줄 바꿈은 한글에서 확인한다. PDF 는 한글에서 다시 내보낸다.
"""
from __future__ import annotations
import argparse, difflib, html, os, re, struct, subprocess, sys, zipfile

MD = "paper/본논문_재작성.md"
HWPX = "논문 최종본.hwpx"
SECTIONS = ("Contents/section0.xml", "Contents/section1.xml", "Contents/section2.xml")
SPACER = re.compile(r'<hp:p [^>]*><hp:run charPrIDRef="\d+"/><hp:linesegarray>.*?</hp:linesegarray></hp:p>')


def plain(line: str) -> str:
    """md 한 줄 -> hwpx 에 들어 있는 글자 (이스케이프와 강조 기호 제거)."""
    s = line.strip()
    s = re.sub(r"^#+\s*", "", s)
    s = s.replace("\\*", "*").replace("\\~", "~").replace("\\_", "_")
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    return s


def md_lines(text: str) -> list[str]:
    return [l for l in text.splitlines() if l.strip() and l.strip() != "---"]


def kind(line: str) -> str:
    l = line.strip()
    if l.startswith("|"):
        return "table"
    m = re.match(r"^(#+)\s", l)
    if m:
        return f"h{len(m.group(1))}"
    if l.startswith("*[<그림"):
        return "figure"
    if l.startswith("<표") or l.startswith("<그림"):
        return "caption" if re.match(r"^<(표|그림) [\d-]+>\s", l) and not re.match(r"^<(표|그림) [\d-]+>(은|는|과|와|에|을|를)", l) else "body"
    if l.startswith("註:"):
        return "note"
    if l.startswith("出處:"):
        return "source"
    return "body"


def cells(line: str) -> list[str]:
    return [plain(c) for c in line.strip().strip("|").split("|")]


def is_sep(line: str) -> bool:
    return bool(re.match(r"^\|\s*:?-{3,}", line.strip()))


def esc(s: str) -> str:
    return html.escape(s, quote=False)


def find_t(xml: dict, text: str, contains: bool = False) -> list[tuple[str, int, int]]:
    """<hp:t>글</hp:t> 의 글이 text 와 같은(contains 면 text 를 품은) 곳 (파일, 글 시작, 글 끝)."""
    hits = []
    target = esc(text)
    for name, s in xml.items():
        for m in re.finditer(r"<hp:t>([^<]*)</hp:t>", s):
            g = m.group(1)
            if (target in g) if contains else (g == target or g.strip() == target):
                hits.append((name, m.start(1), m.end(1)))
    return hits


def para_end(s: str, pos: int) -> int:
    return s.find("</hp:p>", pos) + len("</hp:p>")


# ------------------------------------------------------------ 문단·표 만들기
class Builder:
    def __init__(self, xml: dict, old_lines: list[str]):
        self.xml = xml
        self.tpl: dict[str, tuple[str, str, bool]] = {}
        # 본문 모양은 표지·요약문이 아닌 본문(1장 이후)에서, 제목 모양은 번호가 붙은 제목에서 가져온다
        start = next((i for i, l in enumerate(old_lines) if re.match(r"^##\s+1\.", l.strip())), 0)
        for line in old_lines[start:]:
            k = kind(line)
            if k in self.tpl or k in ("table", "figure"):
                continue
            if k.startswith("h") and not re.match(r"^\d+(\.\d+)*\.\s", plain(line)):
                continue
            h = find_t(xml, plain(line))
            if len(h) != 1:
                continue
            name, ts, _ = h[0]
            s = xml[name]
            head = re.match(r"<hp:p [^>]*>", s[s.rfind("<hp:p ", 0, ts):]).group(0)
            cp = re.match(r'<hp:run charPrIDRef="(\d+)"', s[s.rfind("<hp:run ", 0, ts):]).group(1)
            lead = re.match(r"<hp:t>( *)", s[ts - len("<hp:t>"):]).group(1)
            self.tpl[k] = (head, cp, bool(lead))
        m = SPACER.search(xml.get("Contents/section1.xml", ""))
        self.spacer = m.group(0) if m else ""
        self.next_id = max(int(x) for s in xml.values() for x in re.findall(r'<hp:tbl id="(\d+)"', s)) + 1
        self.next_z = max(int(x) for s in xml.values() for x in re.findall(r'zOrder="(\d+)"', s)) + 1

    def para(self, k: str, text: str) -> str:
        k = k if k in self.tpl else "body"
        head, cp, lead = self.tpl[k]
        return f'{head}<hp:run charPrIDRef="{cp}"><hp:t>{" " if lead else ""}{esc(text)}</hp:t></hp:run></hp:p>'

    @staticmethod
    def _fill_tr(tr: str, texts: list[str], row: int) -> str:
        out, i = [], 0
        for m in re.finditer(r"<hp:tc .*?</hp:tc>", tr, flags=re.S):
            tc = m.group(0)
            t = texts[i] if i < len(texts) else ""
            p0 = re.search(r"<hp:p [^>]*>", tc).group(0)
            cp = re.search(r'<hp:run charPrIDRef="(\d+)"', tc).group(1)
            body = f'{p0}<hp:run charPrIDRef="{cp}"><hp:t>{esc(t)}</hp:t></hp:run></hp:p>'
            tc = re.sub(r"(<hp:subList [^>]*>).*?(</hp:subList>)", lambda mm: mm.group(1) + body + mm.group(2),
                        tc, count=1, flags=re.S)
            tc = re.sub(r'rowAddr="\d+"', f'rowAddr="{row}"', tc)
            out.append(tc)
            i += 1
        return "<hp:tr>" + "".join(out) + "</hp:tr>"

    def table(self, rows: list[list[str]]) -> str | None:
        """열 수가 같은 원고의 표를 본떠 새 표(담는 문단 포함)를 만든다."""
        ncol = len(rows[0])
        for name in ("Contents/section1.xml",) + tuple(n for n in self.xml if n != "Contents/section1.xml"):
            s = self.xml[name]
            for m in re.finditer(rf'<hp:tbl [^>]*colCnt="{ncol}"[^>]*>', s):
                tb, te = m.start(), s.find("</hp:tbl>", m.start()) + len("</hp:tbl>")
                tbl = s[tb:te]
                if "<hp:tbl " in tbl[1:]:
                    continue
                trs = re.findall(r"<hp:tr>.*?</hp:tr>", tbl, flags=re.S)
                if len(trs) < 2 or any("rowSpan=\"1\"" not in tr or "colSpan=\"1\"" not in tr for tr in trs[:2]):
                    continue
                ps = s.rfind("<hp:p ", 0, tb)
                pe = para_end(s, te)
                head = s[ps:tb]
                tail = s[te:pe]
                new_trs = [self._fill_tr(trs[0] if r == 0 else trs[1], rows[r], r) for r in range(len(rows))]
                h_head = sum(int(x) for x in re.findall(r'<hp:cellSz width="\d+" height="(\d+)"', trs[0])[:1])
                h_body = sum(int(x) for x in re.findall(r'<hp:cellSz width="\d+" height="(\d+)"', trs[1])[:1])
                height = h_head + h_body * (len(rows) - 1)
                open_tag = re.match(r"<hp:tbl [^>]*>", tbl).group(0)
                open_tag = re.sub(r'id="\d+"', f'id="{self.next_id}"', open_tag)
                open_tag = re.sub(r'zOrder="\d+"', f'zOrder="{self.next_z}"', open_tag)
                open_tag = re.sub(r'rowCnt="\d+"', f'rowCnt="{len(rows)}"', open_tag)
                self.next_id += 1; self.next_z += 1
                inner = tbl[len(re.match(r"<hp:tbl [^>]*>", tbl).group(0)):tbl.find("<hp:tr>")]
                inner = re.sub(r'(<hp:sz width="\d+" widthRelTo="\w+" height=")\d+', rf"\g<1>{height}", inner, count=1)
                return head + open_tag + inner + "".join(new_trs) + "</hp:tbl>" + tail
        return None

    def block(self, lines: list[str], after_spacer: bool = False) -> str:
        """원고의 관례대로 제목과 표 제목 앞, 出處 뒤에 빈 문단을 둔다."""
        out, i, sp = [], 0, after_spacer
        while i < len(lines):
            k = kind(lines[i])
            if k == "table":
                j = i
                while j < len(lines) and kind(lines[j]) == "table":
                    j += 1
                rows = [cells(l) for l in lines[i:j] if not is_sep(l)]
                t = self.table(rows)
                if t is None:
                    raise ValueError(f"{len(rows[0])}열 표의 본보기를 찾지 못함")
                out.append(t)
                i, sp = j, False
                continue
            if (k.startswith("h") or k == "caption") and self.spacer and not sp:
                out.append(self.spacer)
            out.append(self.para(k, plain(lines[i])))
            sp = False
            if k == "source" and self.spacer and i + 1 < len(lines):
                out.append(self.spacer)
                sp = True
            i += 1
        return "".join(out)


# ------------------------------------------------------------ 찾기
def locate_para_end(xml: dict, line: str) -> tuple[str, int] | None:
    """md 한 줄에 해당하는 문단의 끝 위치. 글 전체가 하나의 <hp:t> 가 아니면 끝 30자로 찾는다."""
    if kind(line) == "table":
        texts = sorted(cells(line), key=len, reverse=True)
        for t in texts:
            h = find_t(xml, t)
            if len(h) == 1:
                name, ts, _ = h[0]
                s = xml[name]
                return name, s.find("</hp:tr>", ts) + len("</hp:tr>")
        return None
    text = plain(line)
    h = find_t(xml, text)
    if len(h) != 1:
        h = find_t(xml, text[-30:], contains=True)
    if len(h) != 1:
        return None
    name, ts, _ = h[0]
    s = xml[name]
    if s.rfind("<hp:tc ", 0, ts) > s.rfind("</hp:tc>", 0, ts):      # 표 칸 안
        return None
    return name, para_end(s, ts)


def substring_edit(xml: dict, old: str, new: str):
    """각주로 나뉜 문단: 바뀐 부분과 앞뒤 글자를 담은 <hp:t> 하나를 찾아 그 부분만 바꾼다."""
    p = 0
    while p < min(len(old), len(new)) and old[p] == new[p]:
        p += 1
    q = 0
    while q < min(len(old), len(new)) - p and old[-1 - q] == new[-1 - q]:
        q += 1
    om, nm = old[p:len(old) - q], new[p:len(new) - q]
    for ctx in (40, 25, 15, 8, 4, 0):
        left = old[max(0, p - ctx):p]
        right = old[len(old) - q:len(old) - q + ctx]
        if not (left or right or om):
            continue
        h = find_t(xml, left + om + right, contains=True)
        if len(h) == 1:
            name, ts, te = h[0]
            seg = xml[name][ts:te]
            k = seg.find(esc(left + om + right))
            a = ts + k + len(esc(left))
            return name, a, a + len(esc(om)), nm
    return None


# ------------------------------------------------------------ 그림
def png_size(b: bytes) -> tuple[int, int]:
    return struct.unpack(">II", b[16:24])


def main():
    ap = argparse.ArgumentParser(description="md 원고 수정 -> hwpx")
    ap.add_argument("--base", default="HEAD", help="비교할 옛 md 의 git 리비전")
    ap.add_argument("--md", default=MD)
    ap.add_argument("--hwpx", default=HWPX)
    ap.add_argument("--out", default=None, help="저장 경로 (기본: 원본 덮어쓰기)")
    ap.add_argument("--image", nargs="*", default=[], help="그림 바꾸기: imageN=경로.png")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    old = subprocess.run(["git", "show", f"{a.base}:{a.md}"], capture_output=True, check=True).stdout.decode("utf-8")
    new = open(a.md, encoding="utf-8").read()
    ol, nl = md_lines(old), md_lines(new)

    z = zipfile.ZipFile(a.hwpx)
    infos = z.infolist()
    data = {i.filename: z.read(i.filename) for i in infos}
    xml = {n: data[n].decode("utf-8") for n in SECTIONS if n in data}
    B = Builder(xml, ol)

    edits, fails = [], []          # (파일, 시작, 끝, 넣을 XML, 설명)

    def text_edit(lo: str, ln: str):
        h = find_t(xml, plain(lo))
        if len(h) == 1:
            name, ts, te = h[0]
            lead = " " if xml[name][ts:te].startswith(" ") else ""
            edits.append((name, ts, te, lead + esc(plain(ln)), f"바꿈: {plain(lo)[:50]}…"))
            return
        r = substring_edit(xml, plain(lo), plain(ln))
        if r:
            name, s0, s1, nm = r
            edits.append((name, s0, s1, esc(nm), f"부분 바꿈: {nm[:50]}…"))
        else:
            fails.append(f"문단({len(h)}곳): {plain(lo)[:70]}")

    def insert_after(anchor: str | None, lines: list[str]):
        loc = locate_para_end(xml, anchor) if anchor else None
        if loc is None:
            fails.append(f"새 문단의 위치: {plain(lines[0])[:70]}")
            return
        name, pos = loc
        if kind(anchor) == "table" and all(kind(l) == "table" for l in lines):
            # 기존 표에 행 더하기
            s = xml[name]
            tr_s = s.rfind("<hp:tr>", 0, pos)
            tr = s[tr_s:pos]
            row0 = int(re.search(r'rowAddr="(\d+)"', tr).group(1))
            rows = [cells(l) for l in lines if not is_sep(l)]
            new_trs = "".join(Builder._fill_tr(tr, r, row0 + 1 + i) for i, r in enumerate(rows))
            tb = s.rfind("<hp:tbl ", 0, pos)
            open_tag = re.match(r"<hp:tbl [^>]*>", s[tb:]).group(0)
            n0 = int(re.search(r'rowCnt="(\d+)"', open_tag).group(1))
            h = int(re.search(r'<hp:cellSz width="\d+" height="(\d+)"', tr).group(1))
            sz = re.search(r'<hp:sz width="\d+" widthRelTo="\w+" height="(\d+)"', s[tb:])
            edits.append((name, tb, tb + len(open_tag), re.sub(r'rowCnt="\d+"', f'rowCnt="{n0 + len(rows)}"', open_tag),
                          "표 행 수"))
            edits.append((name, tb + sz.start(1), tb + sz.end(1), str(int(sz.group(1)) + h * len(rows)), "표 높이"))
            edits.append((name, pos, pos, new_trs, f"표에 행 {len(rows)}개: {rows[0][0][:30]}…"))
            return
        try:
            x = B.block(lines)
        except ValueError as ex:
            fails.append(f"{ex}: {plain(lines[0])[:60]}")
            return
        edits.append((name, pos, pos, x, f"새 문단 {len(lines)}줄: {plain(lines[0])[:50]}…"))

    sm = difflib.SequenceMatcher(a=ol, b=nl, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        olds, news = ol[i1:i2], nl[j1:j2]
        # 표: 같은 수의 행이 바뀐 경우 칸 단위로
        if olds and all(kind(l) == "table" for l in olds + news) and len(olds) == len(news):
            for lo, ln in zip(olds, news):
                for co, cn in zip(cells(lo), cells(ln)):
                    if co != cn:
                        h = find_t(xml, co)
                        if len(h) == 1:
                            edits.append((*h[0], esc(cn), f"표 칸: {co[:40]}… -> {cn[:40]}…"))
                        else:
                            fails.append(f"표 칸({len(h)}곳): {co[:60]}")
            continue
        # 문단 바꾸기: 종류가 같은 줄끼리 앞에서부터 짝짓는다
        k = 0
        while k < min(len(olds), len(news)) and kind(olds[k]) == kind(news[k]) and kind(olds[k]) != "table":
            k += 1
        for lo, ln in zip(olds[:k], news[:k]):
            text_edit(lo, ln)
        if len(news) > k:
            anchor = olds[k - 1] if k > 0 else (ol[i1 - 1] if i1 > 0 else None)
            insert_after(anchor, news[k:])
        if len(olds) > k:
            fails.append(f"지운 문단(직접 지울 것): {plain(olds[k])[:70]}")

    # 그림 바꾸기
    for spec in a.image:
        nm, path = spec.split("=", 1)
        b = open(path, "rb").read()
        w, h = png_size(b)
        data[f"BinData/{nm}.png"] = b
        for name, s in xml.items():
            i = s.find(f'binaryItemIDRef="{nm}"')
            if i < 0:
                continue
            ps, pe = s.rfind("<hp:pic", 0, i), s.find("</hp:pic>", i)
            pic = s[ps:pe]
            pic = re.sub(r'<hp:imgDim dimwidth="\d+" dimheight="\d+"/>',
                         f'<hp:imgDim dimwidth="{w * 75}" dimheight="{h * 75}"/>', pic)
            pic = re.sub(r'<hp:imgClip left="0" right="\d+" top="0" bottom="\d+"/>',
                         f'<hp:imgClip left="0" right="{w * 75}" top="0" bottom="{h * 75}"/>', pic)
            edits.append((name, ps, pe, pic, f"그림 {nm}: {path} ({w}×{h})"))

    for e in edits:
        print("적용:", e[4])
    for f in fails:
        print("미적용:", f)
    if a.dry_run:
        return

    # 뒤에서부터 적용해 위치가 어긋나지 않게 한다
    for name in xml:
        s = xml[name]
        prev = None
        for fn, ts, te, x, _ in sorted([e for e in edits if e[0] == name], key=lambda e: (-e[1], -e[2])):
            assert prev is None or te <= prev, "수정 위치가 겹칩니다"
            s = s[:ts] + x + s[te:]
            prev = ts
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
