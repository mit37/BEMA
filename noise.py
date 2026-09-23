"""Character-level typo perturbations used by the robustness tests."""
import string

# Approximate QWERTY neighbors: same-index keys in adjacent rows ignore the
# physical row stagger, so a few "neighbors" are one key off.
KEYBOARD_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
NEIGHBORS = {}
for r, row in enumerate(KEYBOARD_ROWS):
    for c, ch in enumerate(row):
        near = set()
        for dr in (-1, 0, 1):
            rr = r + dr
            if 0 <= rr < len(KEYBOARD_ROWS):
                for dc in (-1, 0, 1):
                    cc = c + dc
                    if 0 <= cc < len(KEYBOARD_ROWS[rr]) and (dr, dc) != (0, 0):
                        near.add(KEYBOARD_ROWS[rr][cc])
        NEIGHBORS[ch] = sorted(near)


def letter_positions(chars):
    return [i for i, ch in enumerate(chars) if ch.lower() in NEIGHBORS]


def swap(chars, rnd):
    if len(chars) >= 3:
        i = rnd.randint(0, len(chars) - 2)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]


def keyboard_sub(chars, rnd):
    pos = letter_positions(chars)
    if pos:
        i = rnd.choice(pos)
        repl = rnd.choice(NEIGHBORS[chars[i].lower()])
        chars[i] = repl.upper() if chars[i].isupper() else repl


def delete(chars, rnd):
    pos = letter_positions(chars)
    if len(pos) > 1:
        del chars[rnd.choice(pos)]


def insert(chars, rnd):
    chars.insert(rnd.randint(0, len(chars)), rnd.choice(string.ascii_lowercase))


NOISE = {
    "swap2": [swap, swap],
    "swap4": [swap, swap, swap, swap],
    "keyboard2": [keyboard_sub, keyboard_sub],
    "delete2": [delete, delete],
    "insert2": [insert, insert],
    "mixed3": [keyboard_sub, delete, insert],
}
HELD_OUT = {"keyboard2", "delete2", "insert2", "mixed3"}


def perturb(text, ops, rnd):
    chars = list(text)
    for op in ops:
        op(chars, rnd)
    return "".join(chars)
