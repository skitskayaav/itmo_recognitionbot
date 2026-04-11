"""
Парсер OMML-матриц из Word .docx для корректного отображения в Telegram.
"""
import re

_MATH_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
_WORD_NS  = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _parse_matrix_elem(mat_elem):
    """Рекурсивно разбирает m:m в list[list[str]], поддерживает блочные матрицы."""
    rows_elems = mat_elem.findall(f"{{{_MATH_NS}}}mr")
    if not rows_elems:
        return []

    # Проверяем, есть ли блок-ячейки
    has_blocks = any(
        cell.find(f".//{{{_MATH_NS}}}m") is not None
        for tr in rows_elems
        for cell in tr.findall(f"{{{_MATH_NS}}}e")
    )

    if not has_blocks:
        result = []
        for tr in rows_elems:
            row = []
            for cell in tr.findall(f"{{{_MATH_NS}}}e"):
                texts = cell.findall(f".//{{{_MATH_NS}}}t")
                raw = "".join(t.text or "" for t in texts).strip()
                parts = re.findall(r"-?\d+", raw)
                row.extend(parts) if len(parts) > 1 else row.append(raw)
            result.append(row)
        return result

    # Блочная матрица — сшиваем
    result = []
    for tr in rows_elems:
        cells = tr.findall(f"{{{_MATH_NS}}}e")
        block_row_data = []  # list of sub-matrices for this block-row
        for cell in cells:
            sub = cell.find(f"{{{_MATH_NS}}}m")
            if sub is not None:
                block_row_data.append(_parse_matrix_elem(sub))
            else:
                texts = cell.findall(f".//{{{_MATH_NS}}}t")
                raw = "".join(t.text or "" for t in texts).strip()
                parts = re.findall(r"-?\d+", raw)
                block_row_data.append([[p] for p in parts] if parts else [[raw]])

        n_inner = max((len(b) for b in block_row_data), default=0)
        for ri in range(n_inner):
            row = []
            for block in block_row_data:
                if ri < len(block):
                    row.extend(block[ri])
                else:
                    row.extend([""] * (len(block[0]) if block else 1))
            result.append(row)

    return result


def format_matrix(rows):
    """Форматирует матрицу псевдографикой для Telegram (monospace)."""
    rows = [r for r in rows if r]
    if not rows:
        return ""
    n_cols = max(len(r) for r in rows)
    for r in rows:
        while len(r) < n_cols:
            r.append("")
    col_w = [max(len(rows[ri][ci]) for ri in range(len(rows))) for ci in range(n_cols)]
    n = len(rows)
    lines = []
    for ri, row in enumerate(rows):
        cells = "  ".join(row[ci].rjust(col_w[ci]) for ci in range(len(row)))
        if n == 1:
            bl, br = "(", ")"
        elif ri == 0:
            bl, br = "⎡", "⎤"
        elif ri == n - 1:
            bl, br = "⎣", "⎦"
        else:
            bl, br = "⎢", "⎥"
        lines.append(f"{bl} {cells} {br}")
    return "\n".join(lines)


def extract_omath_text(om_elem):
    """Конвертирует m:oMath в строку. Матрицы — блочно, формулы — инлайн."""
    mat = om_elem.find(f".//{{{_MATH_NS}}}m")
    if mat is not None:
        rows = _parse_matrix_elem(mat)
        if rows:
            return format_matrix(rows)
    texts = om_elem.findall(f".//{{{_MATH_NS}}}t")
    return " ".join(t.text.strip() for t in texts if t.text and t.text.strip())


def extract_paragraph_full_text(p):
    """
    Извлекает полный текст параграфа с формулами и матрицами.
    Матрицы вставляются как отдельные блоки с переносами строк.
    Формулы — инлайн.
    """
    parts = []
    for child in p._element:
        local = child.tag.split("}")[-1] if "}" in child.tag else child.tag

        if local == "r":
            for t in child.findall(f".//{{{_WORD_NS}}}t"):
                parts.append(t.text or "")

        elif local == "oMathPara":
            for om in child.findall(f"{{{_MATH_NS}}}oMath"):
                rendered = extract_omath_text(om)
                if rendered:
                    parts.append(f"\n{rendered}\n")

        elif local == "oMath":
            rendered = extract_omath_text(child)
            if rendered:
                if "\n" in rendered:
                    parts.append(f"\n{rendered}\n")
                else:
                    parts.append(f" {rendered} ")

        elif local == "hyperlink":
            for t in child.findall(f".//{{{_WORD_NS}}}t"):
                parts.append(t.text or "")

    return "".join(parts)