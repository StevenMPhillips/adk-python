from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import tempfile
import urllib.request


DATASET_URLS = {
    'longmemeval_oracle.json': (
        'https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/'
        'resolve/main/longmemeval_oracle.json?download=true'
    ),
    'longmemeval_s_cleaned.json': (
        'https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/'
        'resolve/main/longmemeval_s_cleaned.json?download=true'
    ),
    'longmemeval_m_cleaned.json': (
        'https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/'
        'resolve/main/longmemeval_m_cleaned.json?download=true'
    ),
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description='Download LongMemEval cleaned dataset files.'
  )
  parser.add_argument(
      '--out',
      type=pathlib.Path,
      default=pathlib.Path('tools/longmemeval/data'),
      help='Output directory for dataset json files.',
  )
  parser.add_argument(
      '--force',
      action='store_true',
      help='Re-download files even if they already exist.',
  )
  return parser.parse_args()


def _download_file(url: str, destination: pathlib.Path) -> None:
  request = urllib.request.Request(
      url,
      headers={
          'User-Agent': 'adk-longmemeval-downloader/1.0',
      },
  )
  with tempfile.NamedTemporaryFile(
      delete=False,
      suffix='.part',
      dir=destination.parent,
  ) as tmp_file:
    temp_path = pathlib.Path(tmp_file.name)

  try:
    with urllib.request.urlopen(request, timeout=120) as response:
      with temp_path.open('wb') as out_file:
        while True:
          chunk = response.read(1_048_576)
          if not chunk:
            break
          out_file.write(chunk)
    temp_path.replace(destination)
  finally:
    if temp_path.exists():
      temp_path.unlink()


def main() -> None:
  args = parse_args()
  out_dir = args.out.resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  downloaded_at_utc = datetime.datetime.now(
      datetime.timezone.utc
  ).isoformat()
  files_metadata: dict[str, dict[str, object]] = {}

  for filename, url in DATASET_URLS.items():
    target_path = out_dir / filename
    skipped = target_path.exists() and not args.force
    if not skipped:
      _download_file(url, target_path)

    files_metadata[filename] = {
        'url': url,
        'path': str(target_path),
        'size_bytes': target_path.stat().st_size,
        'skipped_existing': skipped,
    }

  metadata = {
      'dataset': 'xiaowu0162/longmemeval-cleaned',
      'downloaded_at_utc': downloaded_at_utc,
      'out_dir': str(out_dir),
      'force': args.force,
      'files': files_metadata,
  }
  metadata_path = out_dir / 'metadata.json'
  metadata_path.write_text(
      json.dumps(metadata, indent=2, sort_keys=True),
      encoding='utf-8',
  )
  print(f'Wrote metadata: {metadata_path}')


if __name__ == '__main__':
  main()
