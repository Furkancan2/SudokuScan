"""Stage 6 - Validity check of the recognised givens."""


def find_conflicts(grid):
    """Return the set of (row, col) cells that break a sudoku rule:
    the same digit twice in a row, a column or a 3x3 box. 0 = empty."""
    conflicts = set()

    def check(cells):
        seen = {}
        for r, c in cells:
            d = grid[r][c]
            if d == 0:
                continue
            if d in seen:
                conflicts.add((r, c))
                conflicts.add(seen[d])
            else:
                seen[d] = (r, c)

    for i in range(9):
        check([(i, c) for c in range(9)])                  # row i
        check([(r, i) for r in range(9)])                  # column i
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            check([(br + r, bc + c) for r in range(3) for c in range(3)])
    return conflicts


def verdict(grid):
    conflicts = find_conflicts(grid)
    return ("valid" if not conflicts else "invalid"), sorted(conflicts)
