import sys
from jiwer import cer, wer

def read_file(path: str) -> str:
	with open(path, "r", encoding="utf-8") as f:
		return f.read()


def print_metrics(ref_path: str, hyp_path: str) -> None:
	ref_text = read_file(ref_path)
	hyp_text = read_file(hyp_path)
	wer_res = wer(ref_text, hyp_text)
	cer_res = cer(ref_text, hyp_text)

	print(f"Metrics for files: {ref_path} vs {hyp_path}")

	print(f"WER: {wer_res}")
	# print(f"Original words (N): {wer_res['N']}")
	# print(f"Matched words: {wer_res['matches']}")
	# print(f"Details: S={wer_res['substitutions']}, D={wer_res['deletions']}, I={wer_res['insertions']}")
	print(f"CER: {cer_res}")
	# print(f"Original characters (N): {cer_res['N_chars']}")
	# print(f"Matched characters: {cer_res['matches']}")
	# print(f"Chars details: S={cer_res['substitutions']}, D={cer_res['deletions']}, I={cer_res['insertions']}")


if __name__ == "__main__":
	print_metrics(sys.argv[1], sys.argv[2])
