# Mini GPT from Scratch

## Türkçe

Bu depo, karakter düzeyinde metin üreten küçük bir GPT/Transformer modelini adım adım yeniden kurma çalışmasıdır. Çalışma, Andrej Karpathy'nin [*Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY) videosundan ilham alır. Amaç hazır `nn.Transformer` veya `nn.MultiheadAttention` kullanmak değil; bigram modeli, causal attention, çoklu head ve Transformer bloklarını PyTorch'un temel parçalarıyla görünür hâle getirmektir.

Bu, videonun resmî kod deposu ya da birebir kopyası değildir; öğrenme amacıyla kendi başıma yeniden kurduğum bir uygulamadır.

## Nasıl düzenlendi?

Öğrenme sürecinin ana kaynağı 12 adet Jupyter notebook'tur. Her notebookta önce fikir küçük örneklerle incelendi, ardından o adımdaki kod ortak Python modüllerine aktarıldı. Bu yüzden aşağıdaki sıra hem notebookların hem de `.py` dosyalarındaki eklemelerin sırasıdır: sonraki her adım, öncekilerin üzerine inşa edilir.


## Yeniden kurma adımları

| Adım | Konu | Ortak Python koduna eklenenler |
| --- | --- | --- |
| 1 | Karakter verisi ve tokenlaştırma | `data.py`: `build_vocabulary`, `encode`, `decode`, `text_to_tensor` |
| 2 | Shift edilmiş hedefler, train/validation ayrımı ve batch'ler | `data.py`: `split_train_validation`, `get_batch` |
| 3 | Embedding tabanlı bigram dil modeli, loss ve üretim | `bigram.py`: `BigramLanguageModel` |
| 4 | Eğitim adımı ve train/validation loss değerlendirmesi | `training.py`: `train_steps`, `evaluate` |
| 5 | Attention öncesi: geçmiş tokenların causal ortalaması | `attention.py`: `causal_mean_loop`, `causal_mean_weights`, `causal_softmax_weights`, `causal_mean_matmul` |
| 6 | Query, key, value, skorlar, ölçekleme ve softmax | `attention.py`: `scaled_attention_scores`, `scaled_dot_attention_scores` |
| 7 | Causal mask, geleceğe bakmama ve attention dropout | `attention.py`: `CausalSelfAttentionHead` |
| 8 | Token + positional embedding ile tek attention head'li dil modeli | `model.py`: `SingleHeadAttentionLeanguageModel` |
| 9 | Birden fazla attention head, birleştirme ve projection | `attention.py`: `MultiHeadAttention` |
| 10 | Feed-forward ağ, pre-norm ve residual bağlantılar | `model.py`: `FeedForward`, `TransformerBlock` |
| 11 | Transformer bloklarını istifleme, final LayerNorm ve eğitim | `model.py`: `StackedTransformerLanguageModel` |
| 12 | Bağlamı `block_size` ile kırparak üretim ve bigram/Transformer karşılaştırması | `model.py`: `StackedTransformerLanguageModel.generate`; `main.py`: iki modeli eğitme ve örnek metin üretme |

## Dosyalar

```text
.
├── data.py       # Tokenlaştırma, veri bölme ve batch üretme
├── bigram.py     # Baseline bigram dil modeli
├── attention.py  # Causal attention'ın temel yapı taşları
├── model.py      # Tek head ve istiflenmiş Transformer dil modelleri
├── training.py   # Eğitim ve değerlendirme yardımcıları
├── main.py       # Bigram ve mini-GPT eğitim/üretim karşılaştırması
└── input.txt     # Karakter düzeyinde eğitim verisi
```

## Çalıştırma

Python ve PyTorch kurulu bir ortamda depo kökünden:

```bash
pip install torch
python main.py
```

`main.py`, aynı veri üzerinde önce `BigramLanguageModel`, sonra `StackedTransformerLanguageModel` eğitir; eğitim öncesi/sonrası loss değerlerini ve aynı başlangıç promptu için iki üretimi yazdırır. Daha hızlı bir deneme için dosyanın başındaki `STEPS` ve `EVAL_ITERS` sabitlerini küçültebilirsin.

## Temel fikir

Bigram modelinde bir sonraki karakter yalnızca mevcut karaktere bağlıdır. Mini-GPT ise causal self-attention sayesinde, her konumda kendisinden önceki karakterlerin tamamından bağlama göre bilgi alabilir. Bu projenin esas hedefi, bu farkı küçük ve okunabilir bir kod tabanında adım adım görmek.

---

## English

This repository is a step-by-step reconstruction of a small character-level GPT/Transformer that generates text. It is inspired by Andrej Karpathy's [*Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY). Rather than using ready-made `nn.Transformer` or `nn.MultiheadAttention` modules, the project makes the bigram baseline, causal attention, multi-head attention, and Transformer blocks explicit with basic PyTorch building blocks.

This is not the official repository for the video or a line-by-line copy of it. It is an independent implementation built for learning.

### How the project is organized

The learning path consists of twelve Jupyter notebooks. Each notebook introduces an idea with small examples, then demonstrates the corresponding code in the shared Python modules. The steps are cumulative: every later notebook builds on the work of the earlier ones.


### Reconstruction path

| Step | Topic | Python code introduced at this stage |
| --- | --- | --- |
| 1 | Character data and tokenization | `data.py`: `build_vocabulary`, `encode`, `decode`, `text_to_tensor` |
| 2 | Shifted targets, train/validation split, and batches | `data.py`: `split_train_validation`, `get_batch` |
| 3 | An embedding-based bigram language model, loss, and generation | `bigram.py`: `BigramLanguageModel` |
| 4 | Training steps and train/validation loss evaluation | `training.py`: `train_steps`, `evaluate` |
| 5 | Causal averaging before attention | `attention.py`: `causal_mean_loop`, `causal_mean_weights`, `causal_softmax_weights`, `causal_mean_matmul` |
| 6 | Query, key, value, scores, scaling, and softmax | `attention.py`: `scaled_attention_scores`, `scaled_dot_attention_scores` |
| 7 | Causal masks, no future access, and attention dropout | `attention.py`: `CausalSelfAttentionHead` |
| 8 | Token and positional embeddings with one attention head | `model.py`: `SingleHeadAttentionLeanguageModel` |
| 9 | Parallel attention heads, concatenation, and projection | `attention.py`: `MultiHeadAttention` |
| 10 | Feed-forward networks, residual connections, and pre-norm Transformer blocks | `model.py`: `FeedForward`, `TransformerBlock` |
| 11 | Stacking blocks, final LayerNorm, and training | `model.py`: `StackedTransformerLanguageModel` |
| 12 | Context cropping, generation, and bigram/Transformer diagnosis | `model.py`: `StackedTransformerLanguageModel.generate`; `main.py`: training and sampling both models |

### Files

```text
.
├── data.py       # Tokenization, data splitting, and batch creation
├── bigram.py     # The baseline bigram language model
├── attention.py  # The basic building blocks of causal attention
├── model.py      # Single-head and stacked Transformer language models
├── training.py   # Training and evaluation helpers
├── notebooks/    # Clean English companion notebooks for the 12 steps
├── main.py       # Bigram vs. mini-GPT training and generation comparison
└── input.txt     # Character-level training data
```

### Run the project

With Python and PyTorch installed, run the following from the repository root:

```bash
pip install torch
python main.py
```

`main.py` trains `BigramLanguageModel` and `StackedTransformerLanguageModel` on the same data, prints loss values before and after training, and samples text from both models with the same initial prompt. For a quicker smoke test, reduce `STEPS` and `EVAL_ITERS` near the top of `main.py`.

### Core idea

In a bigram model, the next character depends only on the current one. The mini-GPT can use causal self-attention to condition each position on all preceding characters in its context window. The purpose of this project is to make that difference concrete in a small, readable codebase.
