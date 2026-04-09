from opencc_purepy import OpenCC, OpenccConfig

def main():
    text = "“春眠不觉晓，处处闻啼鸟。”"
    # text = "“恶名昭著”"  # Test zho_check() for interference "著"
    opencc = OpenCC()
    opencc.set_config(OpenccConfig.S2T)
    converted = opencc.convert(text, punctuation=True)

    print("Conversion config: {}".format(opencc.get_config()))
    print("Input text: {}".format(text))
    print("Input Chinese text code: {}".format(opencc.zho_check(text)))
    print("Converted text: {}".format(converted))
    print("Converted Chinese text code: {}".format(opencc.zho_check(converted)))

if __name__ == "__main__":
    main()
