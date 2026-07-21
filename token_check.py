python -c "
import os, tiktoken
encoder = tiktoken.get_encoding('cl100k_base')
total = 0
count = 0
for f in os.listdir('data/papers'):
    if f.endswith('.txt'):
        with open(f'data/papers/{f}') as fh:
            total += len(encoder.encode(fh.read()))
        count += 1
print(f'Files: {count}')
print(f'Total tokens: {total:,}')
print(f'Target: 2,000,000')
print(f'Status: {\"REACHED\" if total >= 2000000 else \"NEED MORE\"}')
"