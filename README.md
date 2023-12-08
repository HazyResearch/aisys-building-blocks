# (WIP) Building Blocks for AI Systems - UNDER CONSTRUCTION

**This repo is under construction!**

This is a (biased) view of great work studying the building blocks of efficient and performant foundation models.
This Github was originally put together as a place to aggregate materials for a [NeurIPS keynote](https://neurips.cc/virtual/2023/invited-talk/73990) - but we're also hoping to highlight great work across ML Systems.
If you think we're missing something, please open an issue or PR!

If you just want to follow along on the major pieces from the talk, check out these resources:
* Data Wrangling
* FlashAttention
* Simplifying S4
* Convolutions
* Synthetics
* Truly Sub-Quadratic

**Courses.** Here are some high-level courses that can be useful for getting your bearings.
This list is biased by Stanford courses, so please reach out if you think of other resources that are helpful!
* [Stanford CS 324 LLMs](https://stanford-cs324.github.io/winter2022/)
* [Stanford CS 324 Advances in Foundation Models](https://stanford-cs324.github.io/winter2023/)
* [Sasha's talk on do we need attention?](https://github.com/srush/do-we-need-attention/blob/main/DoWeNeedAttention.pdf)
* [Stanford CS 229S Systems for Machine Learning](https://cs229s.stanford.edu/fall2023/)
* [MLSys Seminar](https://mlsys.stanford.edu/)
* [Berkeley AI-Sys](https://ucbrise.github.io/cs294-ai-sys-sp22/)
* [MIT CS 6.5940](https://hanlab.mit.edu/courses/2023-fall-65940)

**Table of contents:**
* [Foundation Models for Systems](#foundation-models-for-systems)
* [Hardware-Aware Algorithms](#hardware-aware-algorithms)
* [Quantization, Pruning, and Distillation](#quantization-pruning-distillation)
* [Can We Replace Attention?](#can-we-replace-attention)
* [Synthetics for Language Modeling](#synthetics-for-language-modeling)
* [Truly Sub-Quadratic Models](#truly-sub-quadratic-models)
* [Systems for Inference](#systems-for-inference)
* [High-Throughput](#high-throughput)
* [New Data Types](#new-data-types)

## Foundation Models for Systems
Foundation models are changing the ways that we build systems for classical problems like data cleaning.
[SIGMOD keynote](https://cs.stanford.edu/~chrismre/papers/SIGMOD-Chris-Re-DataCentric-Foundation-Models-KeyNote.pdf) on this topic.

### Blog Posts
* [Data Wrangling with Foundation Models](https://hazyresearch.stanford.edu/blog/2023-01-13-datawrangling)
* [Ask Me Anything: Leveraging Foundation Models for Private & Personalized Systems](https://hazyresearch.stanford.edu/blog/2023-04-18-personalization)

### Papers
* [Holoclean: Holistic Data Repairs with Probabilistic Inference](https://arxiv.org/abs/1702.00820)
* [Can Foundation Models Wrangle Your Data?](https://arxiv.org/abs/2205.09911)

## Hardware-Aware Algorithms

Hardware-aware algorithms for today's ML primitives, like attention and long convolutions.

### Blog Posts
* [FlashAttention](https://crfm.stanford.edu/2023/01/13/flashattention.html)
* [FlashFFTConv](https://hazyresearch.stanford.edu/blog/2023-11-13-flashfftconv)

### Papers
* [FlashAttention](https://arxiv.org/abs/2205.14135) and [FlashAttention-2](https://arxiv.org/abs/2307.08691)
* [FlashFFTConv: Efficient Convolutions for Long Sequences with Tensor Cores](https://arxiv.org/abs/2311.05908)

## Quantization, Pruning, and Distillation

### Papers
* [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)


## Can We Replace Attention?

Alternatives to attention that scale sub-quadratically in sequence length.
High-level overview of this space: [From Deep to Long Learning](https://hazyresearch.stanford.edu/blog/2023-03-27-long-learning).

### Blog Posts
* [What is a long convolution?]()
* [Simplifying S4](https://hazyresearch.stanford.edu/blog/2022-06-11-simplifying-s4)
* [Sasha's Great Annotated S4](https://srush.github.io/annotated-s4/)
* [H3: Language Modeling with State Space Models and (Almost) No Attention](https://hazyresearch.stanford.edu/blog/2023-01-20-h3)
* [Hyena Blog](https://hazyresearch.stanford.edu/blog/2023-06-08-hyena-safari)
* Mamba tweet threads by [Albert](https://twitter.com/_albertgu/status/1731727672286294400) and [Tri](https://twitter.com/tri_dao/status/1731728602230890895)

### Papers
* [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) and [code](https://github.com/state-spaces/mamba)
* [RWKV](https://arxiv.org/abs/2305.13048) and [code](https://github.com/BlinkDL/RWKV-LM)
* [Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396)
* [Long Range Language Modeling via Gated State Spaces](https://arxiv.org/abs/2206.13947)
* [Hungry Hungry Hippos: Towards Language Modeling with State Space Models](https://arxiv.org/abs/2212.14052)
* [Hyena Hierarchy: Towards Larger Convolutional Language Models](https://arxiv.org/abs/2302.10866)
* [Simplified State Space Layers for Sequence Modeling](https://arxiv.org/abs/2208.04933)
* [On the Parameterization and Initialization of Diagonal State Space Models](https://arxiv.org/abs/2206.11893)
* [Mega: Moving Average Equipped Gated Attention](https://arxiv.org/abs/2209.10655)
* [Simple Hardware-Efficient Long Convolutions for Sequence Modeling](https://arxiv.org/abs/2302.06646)
* DSS, RetNet, LRU, MultiRes, CKConv

## Synthetics for Language Modeling
Synthetic tasks like associative recall have been very helpful in designing new architectures.

### Associative Recall
Associative recall has a long history in machine learning. Various works have proposed synthetic formulations of the task to better understand and evaluate model behavior.
* Associative recall papers: Ba et al, Zhang and Zhou 17, Olsson et al 21

### Blog Posts

### Papers

## Truly Sub-Quadratic Models

ML models are quadratic along another dimension -- model width.
Can we develop models that grow sub-quadratically with model width?

### Blog Posts
* Monarch Blog Post
* M2-BERT
* M2 retrieval (hopefully)
* Butterflies are all you need

### Papers
* [Monarch Mixer](https://arxiv.org/abs/2310.12109)
* [Monarch](https://arxiv.org/abs/2204.00595)
* Kaleidoscope, Pixelated Butterfly
* Papers from the blog post

## Systems for Inference
* MQA, GQA, Matformer, TGI, TensorRT
* [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
* [Flash-Decoding for long-context inference](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)
* [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)

## High-Throughput

### Blogposts
* [Batch computing and the coming age of AI systems](https://hazyresearch.stanford.edu/blog/2023-04-12-batch)

### Papers and projects:
* [FlexGen](FlexGen: High-throughput Generative Inference of Large Language Models with a Single GPU)
* [Evaporate: Language Models Enable Simple Systems for Generating Structured Views of Heterogeneous Data Lakes](https://www.vldb.org/pvldb/vol17/p92-arora.pdf)
* [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html)


## New Data Types
* HyenaDNA, find papers that have used it, blog posts, code, EEG, fMRI

