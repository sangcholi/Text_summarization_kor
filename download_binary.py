import os
import gdown

if os.path.exists('kobart_summary/config.json') and os.path.exists('kobart_summary/pytorch_model.bin') :
    print("== Data existed.==")
    pass
else:
    os.system("rm -rf kobary_summary")
    os.system("mkdir kobart_summary")

    # 학습 시킨 모델 config 파일
    url = "https://drive.google.com/u/0/uc?id=1CAWBjV6zU-6udG4kGnanznVagn22GEtD" 
    output = './kobart_summary/config.json'
    print("Download config.json")
    gdown.download(url, output, quiet=False)

    # 학습 시킨 모델 bin 파일
    url = "https://drive.google.com/u/0/uc?id=1-fb9wOXT7u9AZ4gI90lWg444zVDNPDxj" 
    output = './kobart_summary/pytorch_model.bin'
    print("Download pytorch_model.bin")
    gdown.download(url, output, quiet=False)
