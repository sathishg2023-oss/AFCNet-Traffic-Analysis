from argparse import ArgumentParser
from afcnet.data import pair_files
p = ArgumentParser()
p.add_argument('--images', required=True); p.add_argument('--labels', required=True)
a = p.parse_args()
pairs, missing = pair_files(a.images, a.labels)
print(f'Matched pairs: {len(pairs)}')
print(f'Images without matched labels: {len(missing)}')
for image, label in pairs[:10]: print(f'{image}  <->  {label}')
if missing:
    print('First unmatched images:')
    for image in missing[:10]: print(image)
