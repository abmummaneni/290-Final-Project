import json
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(_PROJECT_ROOT, "data", "manifest.json")) as f:
    m = json.load(f)

fmt = '{:<12} {:<55} {:<30} {}'
print(fmt.format('ID', 'Title', 'Author', 'Status'))
print('-' * 110)
for eid, info in sorted(m.items()):
    if info['medium'] == 'Novel' and info['scrape_status'] in ('NEEDS_MANUAL', 'PARTIAL'):
        wc = info.get('word_count_cleaned', 0)
        tag = info['scrape_status'] + (f', {wc}w' if wc else '')
        print(fmt.format(eid, info['title'][:54], info['author'][:29], tag))
