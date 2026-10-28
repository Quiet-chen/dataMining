"""Patch transformers 4.2.2 to work with hf-mirror.com.

Three fixes, all in transformers/file_utils.py:

1. Point the hardcoded HuggingFace URL prefix at the mirror.
2. Resolve the relative ``Location`` header hf-mirror returns on redirect.
3. On a HEAD failure (hf-mirror occasionally returns 429), fall back to an
   already-cached file instead of raising.

Idempotent: running it twice is harmless. Re-run after reinstalling
transformers.
"""

import pathlib

import transformers.file_utils as fu


path = pathlib.Path(fu.__file__)
source = path.read_text()

source = source.replace(
    "https://huggingface.co/{model_id}",
    "https://hf-mirror.com/{model_id}",
)
source = source.replace(
    "from urllib.parse import urlparse",
    "from urllib.parse import urlparse, urljoin",
)
source = source.replace(
    'url_to_download = r.headers["Location"]',
    'url_to_download = urljoin(url, r.headers["Location"])',
)

old_except = """        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            # etag is already None
            pass"""
new_except = old_except + """
        except requests.exceptions.HTTPError as err:
            # hf-mirror rate-limits with 429; a HEAD failure must not prevent
            # using an already-downloaded file from the cache.
            status = err.response.status_code if err.response is not None else None
            if status not in (429, 500, 502, 503, 504):
                raise"""
if old_except in source and "requests.exceptions.HTTPError as err" not in source:
    source = source.replace(old_except, new_except)

path.write_text(source)
print("patched", path)
