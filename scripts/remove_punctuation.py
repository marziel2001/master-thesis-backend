from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path


def strip_punctuation_text(text: str, normalize_whitespace: bool = True) -> str:
    result = ''.join(ch for ch in text if not unicodedata.category(ch).startswith('P'))
    if normalize_whitespace:
        result = re.sub(r'\s+', ' ', result).strip()
    return result


def remove_punctuation_from_file(input_path: str | Path, output_path: str | Path, normalize_whitespace: bool = True) -> Path:
    in_path = Path(input_path)
    out_path = Path(output_path)

    text = in_path.read_text(encoding='utf-8')
    cleaned = strip_punctuation_text(text, normalize_whitespace=normalize_whitespace)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(cleaned, encoding='utf-8')
    return out_path


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_nopunct{input_path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Remove punctuation from input text file and save output file.'
    )
    parser.add_argument('input_file', help='Path to input text file.')
    parser.add_argument(
        '-o',
        '--output',
        help='Path to output file. If omitted, <input_stem>_nopunct<input_suffix> is used.'
    )
    parser.add_argument(
        '--keep-whitespace',
        action='store_true',
        help='Keep original whitespace (do not collapse whitespace into single spaces).'
    )

    args = parser.parse_args()
    input_path = Path(args.input_file)
    output_path = Path(args.output) if args.output else _default_output_path(input_path)

    written_path = remove_punctuation_from_file(
        input_path,
        output_path,
        normalize_whitespace=not args.keep_whitespace,
    )
    print(f'Written: {written_path}')


if __name__ == '__main__':
    main()
