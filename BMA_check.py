import json, base64

j = json.load(open("opencc_purepy/dicts/dictionary_maxlength.json","r",encoding="utf-8"))
bm = base64.b64decode(j["starter_index"]["bmp_mask"])
bc = base64.b64decode(j["starter_index"]["bmp_cap"])

def probe(char):
    cp = ord(char)
    m = int.from_bytes(bm[cp*8:(cp+1)*8], "little")
    c = int.from_bytes(bc[cp*2:(cp+1)*2], "little")
    print(char, hex(cp), m, c)

for ch in "中華發发一人":
    probe(ch)
