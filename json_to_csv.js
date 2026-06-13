const fs = require("fs");
const path = require("path");

const RUNS_DIR = "./outputs/runs";
const OUTPUT_FILE = "./all-results.csv";

const HEADERS = [
  "runFolder",
  "filename",
  "modelName",
  "modelVersion",
  "computeTime",
  "audioDuration",
  "rtf",
  "wer",
  "cer",
];

function getAllJsonFiles(dir) {
  const result = [];

  const items = fs.readdirSync(dir, { withFileTypes: true });

  for (const item of items) {
    const fullPath = path.join(dir, item.name);

    if (item.isDirectory()) {
      result.push(...getAllJsonFiles(fullPath));
    } else if (
      item.isFile() &&
      item.name.startsWith("transcription") &&
      item.name.endsWith(".json")
    ) {
      result.push(fullPath);
    }
  }

  return result;
}

function csvEscape(value) {
  if (value === null || value === undefined) {
    return "";
  }

  const str = String(value);

  if (str.includes(",") || str.includes('"') || str.includes("\n")) {
    return `"${str.replace(/"/g, '""')}"`;
  }

  return str;
}

const rows = [];

const jsonFiles = getAllJsonFiles(RUNS_DIR);

console.log(`Znaleziono ${jsonFiles.length} plików JSON`);

for (const filePath of jsonFiles) {
  try {
    const json = JSON.parse(fs.readFileSync(filePath, "utf8"));

    const runFolder = path.basename(path.dirname(filePath));

    rows.push({
      runFolder,
      filename: json.filename ?? "",
      modelName: json.modelName ?? "",
      modelVersion: json.modelVersion ?? "",
      computeTime: json.computeTime ?? "",
      audioDuration: json.audioDuration ?? "",
      rtf:
        json.computeTime && json.audioDuration
          ? json.computeTime / json.audioDuration
          : "",
      wer: json.wer ?? "",
      cer: json.cer ?? "",
    });
  } catch (err) {
    console.error(`Błąd w pliku ${filePath}:`, err.message);
  }
}

const csv = [
  HEADERS.join(","),
  ...rows.map((row) => HEADERS.map((h) => csvEscape(row[h])).join(",")),
].join("\n");

fs.writeFileSync(OUTPUT_FILE, csv, "utf8");

console.log(`Gotowe. Wyeksportowano ${rows.length} rekordów do ${OUTPUT_FILE}`);
