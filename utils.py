import re
import unicodedata

def clean_garbage_text(text: str) -> str:
    """
    Removes lines containing garbage characters (e.g., Arabic, etc.), preserving
    Vietnamese (Latin + diacritics), numbers, and punctuation.
    """
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        if not line.strip():
            cleaned_lines.append(line)
            continue

        latin_count = 0
        other_count = 0

        for char in line:
            if char.isspace() or char in '.,;:!?()[]{}\'"\'-_+=/\\@#$%^&*<>~`|':
                continue
            try:
                name = unicodedata.name(char, '')
                if 'LATIN' in name or char.isdigit():
                    latin_count += 1
                else:
                    other_count += 1
            except TypeError: # Handle cases where unicodedata.name fails
                other_count += 1


        total = latin_count + other_count
        if total == 0 or latin_count / total >= 0.5:
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)

def replace_md_image_refs(full_text: str, img_name: str, first: str, later: str) -> str:
    pattern = re.compile(r"!\\[[^\\]]*\\]\(" + re.escape(img_name) + r"\")")
    seen = {"n": 0}

    def _repl(_m):
        seen["n"] += 1
        return first if seen["n"] == 1 else later

    return pattern.sub(_repl, full_text)

def collapse_repeated_words(text: str, max_repeat: int = 2) -> str:
    """
    Reduces word repetitions, e.g., 'abc abc abc abc' -> 'abc abc'.
    """
    tokens = text.split()
    out = []
    run_word = None
    run_len = 0
    for w in tokens:
        if w == run_word:
            run_len += 1
        else:
            run_word = w
            run_len = 1
        if run_len <= max_repeat:
            out.append(w)
    return " ".join(out)

def clean_caption(caption: str) -> str:
    """
    Cleans and standardizes an image caption.
    """
    caption = caption.strip()
    # Normalize whitespace
    caption = re.sub(r"\s+", " ", caption)
    # Truncate long captions to avoid model looping
    if len(caption) > 600:
        caption = caption[:600].rsplit(" ", 1)[0] + "..."
    # Reduce word repetitions
    caption = collapse_repeated_words(caption, max_repeat=2)
    # Shorten if unique word ratio is low (heavy repetition)
    words = caption.split()
    if words:
        uniq_ratio = len(set(words)) / max(1, len(words))
        if uniq_ratio < 0.35 and len(words) > 25:
            caption = " ".join(words[:25]) + "..."
    return caption

def shorten_logo_caption(caption: str) -> str:
    """Shortens caption if it appears to be a logo."""
    caption_lower = caption.lower()

    # If 'logo' is present, simplify to "Logo [Name]"
    if 'logo' in caption_lower:
        match = re.search(r'logo\s+(\w+)', caption, re.IGNORECASE)
        if match:
            return f"Logo {match.group(1)}"
        return "Logo"

    # Truncate long captions
    if len(caption) > 100:
        return caption[:100].rsplit(' ', 1)[0] + "..."
    
    return caption