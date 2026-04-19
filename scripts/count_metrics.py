import sys
from jiwer import cer, wer


def read_file(path: str) -> str:
	with open(path, "r", encoding="utf-8") as f:
		return f.read()


def calculate_metrics_from_text(ref_text: str, hyp_text: str) -> dict[str, float]:
	return {
		"wer": wer(ref_text, hyp_text),
		"cer": cer(ref_text, hyp_text),
	}


def calculate_metrics(ref_path: str, hyp_path: str) -> dict[str, float]:
	ref_text = read_file(ref_path)
	hyp_text = read_file(hyp_path)
	return calculate_metrics_from_text(ref_text, hyp_text)


if __name__ == "__main__":
	calculate_metrics(sys.argv[1], sys.argv[2])
