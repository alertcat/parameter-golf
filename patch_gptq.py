import lzma as L, base64 as B, re
import os

PATH = '/workspace/parameter-golf/train_gpt.py'

with open(PATH) as f:
    w = f.read()

m = re.search(r'B\.b85decode\("([^"]+)"\)', w)
src = L.decompress(
    B.b85decode(m.group(1)),
    format=L.FORMAT_RAW,
    filters=[{'id': L.FILTER_LZMA2}]
).decode()

old = "if not t.is_floating_point()or t.numel()<=65536:"
new = "if not t.is_floating_point()or t.numel()<=65536 or'ds_bias'in name:"

assert old in src, 'Pattern not found in source!'
src_fixed = src.replace(old, new, 1)
print('Patch applied: ds_bias will now passthrough as fp16')

fixed_bytes = src_fixed.encode('utf-8')
compressed = L.compress(
    fixed_bytes,
    format=L.FORMAT_RAW,
    filters=[{'id': L.FILTER_LZMA2, 'preset': 9 | L.PRESET_EXTREME}]
)
b85 = B.b85encode(compressed).decode('ascii')
wrapper = 'import lzma as L,base64 as B\nexec(L.decompress(B.b85decode("' + b85 + '"),format=L.FORMAT_RAW,filters=[{"id":L.FILTER_LZMA2}]))\n'

with open(PATH, 'w') as f:
    f.write(wrapper)

print(f'New train_gpt.py size: {os.path.getsize(PATH)} bytes')

# Roundtrip verify
w2 = open(PATH).read()
m2 = re.search(r'B\.b85decode\("([^"]+)"\)', w2)
src2 = L.decompress(
    B.b85decode(m2.group(1)),
    format=L.FORMAT_RAW,
    filters=[{'id': L.FILTER_LZMA2}]
).decode()
assert "ds_bias'in name" in src2, 'Patch did not survive roundtrip!'
print('PATCH OK - safe to restart training')
