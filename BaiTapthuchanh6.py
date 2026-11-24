"""
Phiếu bài tập: Apriori, FP-growth và luật kết hợp
Bài 1: Apriori trên CSDL bài 1 (minsup, minconf nhập từ bàn phím)
Bài 2: FP-growth trên CSDL bài 1 (minsup, minconf nhập từ bàn phím)
Bài 3: FP-growth, minsup = 80%, minconf = 80%
Bài 4: FP-growth, minsup = 80%, minconf = 80%
"""
from itertools import combinations
from math import ceil

# ----------------------------------------------------------------------
# DỮ LIỆU
# ----------------------------------------------------------------------
DB1 = [
    ["A", "B", "D", "E"],
    ["B", "C", "E"],
    ["A", "B", "D", "E"],
    ["A", "B", "C", "E"],
    ["A", "B", "C", "D", "E"],
    ["B", "C", "D", "E"],
]
DB3 = [
    ["a", "q", "c", "d", "m", "p"],
    ["q", "b", "c", "a", "l", "m", "k"],
    ["b", "a"],
    ["b", "c", "k", "p"],
    ["q", "a", "c", "e", "l", "p", "m", "n"],
]
DB4 = [
    ["a", "d", "c", "q", "m", "p"],
    ["d", "b", "c", "a", "h", "m", "k"],
    ["b", "a", "g"],
    ["b", "c", "k", "p"],
    ["d", "a", "c", "e", "h", "p", "m", "g"],
]


def min_count_from(minsup, n):
    """minsup là tỉ lệ (0..1) -> số giao dịch tối thiểu (làm tròn lên)."""
    return ceil(minsup * n - 1e-9)


# ----------------------------------------------------------------------
# APRIORI
# ----------------------------------------------------------------------
def apriori(transactions, minsup):
    """Trả về dict {frozenset: số lần xuất hiện} của các tập phổ biến."""
    trans = [frozenset(t) for t in transactions]
    min_cnt = min_count_from(minsup, len(trans))

    # L1
    counts = {}
    for t in trans:
        for i in t:
            key = frozenset([i])
            counts[key] = counts.get(key, 0) + 1
    Lk = {s: c for s, c in counts.items() if c >= min_cnt}
    frequent = dict(Lk)

    k = 2
    while Lk:
        prev = sorted(tuple(sorted(s)) for s in Lk)
        prev_set = set(prev)
        # Bước kết nối (join): ghép 2 tập (k-1)-phần tử chung k-2 phần tử đầu
        candidates = set()
        for i in range(len(prev)):
            for j in range(i + 1, len(prev)):
                a, b = prev[i], prev[j]
                if a[:-1] == b[:-1]:
                    cand = tuple(sorted(set(a) | set(b)))
                    # Bước cắt tỉa (prune): mọi tập con (k-1) phải phổ biến
                    if all(sub in prev_set for sub in combinations(cand, k - 1)):
                        candidates.add(cand)
        # Đếm độ hỗ trợ
        counts = {c: 0 for c in candidates}
        for t in trans:
            for c in candidates:
                if t.issuperset(c):
                    counts[c] += 1
        Lk = {frozenset(c): n for c, n in counts.items() if n >= min_cnt}
        frequent.update(Lk)
        k += 1
    return frequent


# ----------------------------------------------------------------------
# FP-GROWTH
# ----------------------------------------------------------------------
class FPNode:
    def __init__(self, item, parent):
        self.item = item
        self.count = 0
        self.parent = parent
        self.children = {}
        self.link = None  # node-link


def build_fp_tree(weighted_trans, min_cnt):
    """weighted_trans: list[(list_items, count)]. Trả về (root, header, item_counts)."""
    item_counts = {}
    for items, c in weighted_trans:
        for i in items:
            item_counts[i] = item_counts.get(i, 0) + c
    item_counts = {i: c for i, c in item_counts.items() if c >= min_cnt}
    if not item_counts:
        return None, None, item_counts

    # Thứ tự: giảm dần theo support (hòa thì theo tên cho ổn định)
    order = {i: r for r, i in enumerate(
        sorted(item_counts, key=lambda x: (-item_counts[x], str(x))))}
    root = FPNode(None, None)
    header = {i: None for i in item_counts}  # item -> node đầu tiên

    for items, c in weighted_trans:
        items = sorted((i for i in items if i in item_counts), key=lambda x: order[x])
        node = root
        for i in items:
            if i not in node.children:
                child = FPNode(i, node)
                node.children[i] = child
                # thêm vào node-link
                child.link = header[i]
                header[i] = child
            node = node.children[i]
            node.count += c
    return root, header, item_counts


def fp_growth(transactions, minsup):
    """Trả về dict {frozenset: số lần xuất hiện} của các tập phổ biến."""
    n = len(transactions)
    min_cnt = min_count_from(minsup, n)
    weighted = [(list(t), 1) for t in transactions]
    frequent = {}
    _mine(weighted, min_cnt, frozenset(), frequent)
    return frequent


def _mine(weighted, min_cnt, suffix, frequent):
    root, header, item_counts = build_fp_tree(weighted, min_cnt)
    if root is None:
        return
    # Duyệt các item từ support thấp đến cao
    for item in sorted(item_counts, key=lambda x: (item_counts[x], str(x))):
        new_set = suffix | {item}
        frequent[frozenset(new_set)] = item_counts[item]

        # Conditional pattern base của item
        cond_base = []
        node = header[item]
        while node is not None:
            path = []
            p = node.parent
            while p is not None and p.item is not None:
                path.append(p.item)
                p = p.parent
            if path:
                cond_base.append((path, node.count))
            node = node.link
        _mine(cond_base, min_cnt, frozenset(new_set), frequent)


# ----------------------------------------------------------------------
# SINH LUẬT KẾT HỢP
# ----------------------------------------------------------------------
def gen_rules(frequent, n_trans, minconf):
    """Trả về list (X, Y, support, confidence) với conf >= minconf."""
    rules = []
    for itemset, cnt in frequent.items():
        if len(itemset) < 2:
            continue
        items = sorted(itemset)
        for r in range(1, len(items)):
            for left in combinations(items, r):
                X = frozenset(left)
                Y = itemset - X
                conf = cnt / frequent[X]  # mọi tập con của tập phổ biến cũng phổ biến
                if conf >= minconf - 1e-9:
                    rules.append((X, Y, cnt / n_trans, conf))
    rules.sort(key=lambda r: (-r[3], -r[2], sorted(r[0]), sorted(r[1])))
    return rules


# ----------------------------------------------------------------------
# HIỂN THỊ
# ----------------------------------------------------------------------
def fmt(s):
    return "{" + ", ".join(sorted(s)) + "}"


def run(name, algo, db, minsup, minconf):
    n = len(db)
    print("=" * 70)
    print(f"{name}: {algo.__name__.upper()} | minsup = {minsup:.0%} "
          f"(>= {min_count_from(minsup, n)}/{n} giao dịch) | minconf = {minconf:.0%}")
    print("=" * 70)
    freq = algo(db, minsup)

    print("\nCác tập phổ biến:")
    if not freq:
        print("  (không có)")
    for k in sorted({len(s) for s in freq}):
        print(f"  --- L{k} ---")
        for s, c in sorted(freq.items(), key=lambda x: sorted(x[0])):
            if len(s) == k:
                print(f"  {fmt(s):<22} count = {c}  support = {c / n:.2%}")

    rules = gen_rules(freq, n, minconf)
    print("\nCác luật kết hợp:")
    if not rules:
        print("  (không có luật nào thỏa mãn)")
    for X, Y, sup, conf in rules:
        print(f"  {fmt(X)} => {fmt(Y)}   support = {sup:.2%}, confidence = {conf:.2%}")
    print()


def read_percent(prompt):
    while True:
        try:
            v = float(input(prompt).replace(",", "."))
            if v > 1:  # nhập kiểu phần trăm, ví dụ 50 -> 0.5
                v /= 100
            if 0 < v <= 1:
                return v
        except ValueError:
            pass
        print("  Giá trị không hợp lệ, nhập lại (ví dụ 50 hoặc 0.5).")


def main():
    print("1. Bài 1 - Apriori (CSDL bài 1)")
    print("2. Bài 2 - FP-growth (CSDL bài 1)")
    print("3. Bài 3 - FP-growth (minsup = 80%, minconf = 80%)")
    print("4. Bài 4 - FP-growth (minsup = 80%, minconf = 80%)")
    print("5. Chạy tất cả")
    choice = input("Chọn bài: ").strip()

    if choice in ("1", "2"):
        minsup = read_percent("Nhập độ hỗ trợ tối thiểu (minsup, vd 50 hoặc 0.5): ")
        minconf = read_percent("Nhập độ tin cậy tối thiểu (minconf, vd 70 hoặc 0.7): ")
        if choice == "1":
            run("Bài 1", apriori, DB1, minsup, minconf)
        else:
            run("Bài 2", fp_growth, DB1, minsup, minconf)
    elif choice == "3":
        run("Bài 3", fp_growth, DB3, 0.8, 0.8)
    elif choice == "4":
        run("Bài 4", fp_growth, DB4, 0.8, 0.8)
    elif choice == "5":
        minsup = read_percent("Nhập minsup cho bài 1, 2: ")
        minconf = read_percent("Nhập minconf cho bài 1, 2: ")
        run("Bài 1", apriori, DB1, minsup, minconf)
        run("Bài 2", fp_growth, DB1, minsup, minconf)
        run("Bài 3", fp_growth, DB3, 0.8, 0.8)
        run("Bài 4", fp_growth, DB4, 0.8, 0.8)
    else:
        print("Lựa chọn không hợp lệ.")


if __name__ == "__main__":
    main()