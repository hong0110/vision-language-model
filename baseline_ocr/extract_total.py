import re
import unicodedata

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def extract_total_vn(text):
    text_lower = remove_accents(text.lower())

    # ===== ƯU TIÊN KEYWORD =====
    keywords = [
        "tong cong tien thanh toan",
        "tong tien thanh toan",
        "total payment",
        "tong tien",
        "tong",
        "thanh toan",
    ]

    for kw in keywords:
        idx = text_lower.find(kw)
        if idx != -1:
            segment = text[idx:idx+100]

            nums = re.findall(r"\d[\d.,]+", segment)

            cleaned = []
            for n in nums:
                val = n.replace(",", "").replace(".", "")
                if val.isdigit():
                    v = int(val)

                    # lọc số hợp lý
                    if 1000 <= v <= 100000000:
                        cleaned.append(v)

            if cleaned:
                return max(cleaned)

    # ===== FALLBACK =====
    nums = re.findall(r"\d[\d.,]+", text)

    candidates = []
    for n in nums:
        val = n.replace(",", "").replace(".", "")
        if val.isdigit():
            v = int(val)

            if 1000 <= v <= 100000000:
                candidates.append(v)

    if candidates:
        return max(candidates)

    return None