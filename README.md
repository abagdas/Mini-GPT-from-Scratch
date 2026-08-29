# Mini GPT from Scratch

## Türkçe

Bu depo, karakter düzeyinde metin üreten küçük bir GPT/Transformer uygulamasıdır. Çalışma, Andrej Karpathy'nin [*Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY) videosundan ilham alır.

Hazır `nn.Transformer` veya `nn.MultiheadAttention` kullanmak yerine bigram modeli, causal attention, çoklu attention head ve Transformer blokları temel PyTorch yapı taşlarıyla kurulmuştur. Bu, videonun resmî kod deposu ya da birebir kopyası değildir; öğrenme amacıyla yapılmış bağımsız bir uygulamadır.

### İçerik

- `data.py`: karakter sözlüğü, encode/decode, veri bölme ve rastgele batch üretimi
- `bigram.py`: yalnız mevcut tokena bakan baseline bigram dil modeli
- `attention.py`: causal ortalama, query/key/value attention, causal head ve multi-head attention
- `model.py`: token ve position embeddingleri, Transformer blockları ve mini-GPT modeli
- `training.py`: eğitim ve train/validation değerlendirme yardımcıları
- `main.py`: bigram ve mini-GPT'yi aynı veri üzerinde eğitip üreten çalıştırılabilir örnek

### Çalıştırma

Python ve PyTorch kurulu bir ortamda depo kökünden:

```bash
pip install torch
python main.py
```

`main.py`, aynı veri üzerinde önce `BigramLanguageModel`, sonra `StackedTransformerLanguageModel` eğitir; eğitim öncesi/sonrası loss değerlerini ve aynı başlangıç promptu için iki üretimi yazdırır. Hızlı bir deneme için dosyanın başındaki `STEPS` ve `EVAL_ITERS` sabitlerini küçültebilirsin.

### Temel fikir

Bigram modelinde bir sonraki karakter yalnızca mevcut karaktere bağlıdır. Mini-GPT ise causal self-attention sayesinde, her konumda bağlam penceresindeki önceki karakterlerden bilgi alabilir. Projenin amacı bu farkı küçük ve okunabilir bir kod tabanında görünür kılmaktır.

---

## English

This repository is a small character-level GPT/Transformer implementation that generates text. It is inspired by Andrej Karpathy's [*Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY).

Instead of using ready-made `nn.Transformer` or `nn.MultiheadAttention` modules, the project builds a bigram baseline, causal attention, multi-head attention, and Transformer blocks from basic PyTorch components. It is not the official repository for the video or a line-by-line copy; it is an independent implementation built for learning.

### Contents

- `data.py`: character vocabulary, encode/decode, data splitting, and random batch creation
- `bigram.py`: the baseline bigram language model that sees only the current token
- `attention.py`: causal averaging, query/key/value attention, a causal head, and multi-head attention
- `model.py`: token and positional embeddings, Transformer blocks, and the mini-GPT model
- `training.py`: training and train/validation evaluation helpers
- `main.py`: an executable example that trains and samples both the bigram and mini-GPT models on the same data

### Run the project

With Python and PyTorch installed, run the following from the repository root:

```bash
pip install torch
python main.py
```

`main.py` trains `BigramLanguageModel` and `StackedTransformerLanguageModel` on the same data, prints loss values before and after training, and samples text from both models with the same initial prompt. For a quicker smoke test, reduce `STEPS` and `EVAL_ITERS` near the top of `main.py`.

### Core idea

In a bigram model, the next character depends only on the current one. The mini-GPT can use causal self-attention to condition each position on preceding characters inside its context window. The goal is to make that difference concrete in a small, readable codebase.
