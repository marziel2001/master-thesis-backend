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
	print(f"CER: {cer_res}")


if __name__ == "__main__":
	print_metrics(sys.argv[1], sys.argv[2])
