# (WIP) Building Blocks for AI Systems - UNDER CONSTRUCTION

**This repo is under construction!**

This is a (biased) view of great work studying the building blocks of foundation models.
This Github was originally put together as a place to aggregate materials for a [NeurIPS keynote](https://neurips.cc/virtual/2023/invited-talk/73990) - but we're also hoping to highlight great work across ML Systems.
If you think we're missing something, please open an issue or PR!

If you just want to follow along on the major pieces from the talk, check out these blog posts:
* [Data Wrangling with Foundation Models](https://hazyresearch.stanford.edu/blog/2023-01-13-datawrangling)
* [FlashAttention](https://hazyresearch.stanford.edu/blog/2023-01-12-flashattention-long-sequences) and [FlashAttention-2](https://hazyresearch.stanford.edu/blog/2023-07-17-flash2)
* [Simplifying S4](https://hazyresearch.stanford.edu/blog/2022-06-11-simplifying-s4)
* Long Convolutions for GPT-style Models
* Synthetics
* Truly Sub-Quadratic Models

**Courses.** Here are some high-level courses that can be useful for getting your bearings.
This list is biased by Stanford courses, so please reach out if you think of other resources that are helpful!
* [CS 324 LLMs](https://stanford-cs324.github.io/winter2022/)
* [CS 324 Advances in Foundation Models](https://stanford-cs324.github.io/winter2023/)
* [Sasha's talk on do we need attention?](https://github.com/srush/do-we-need-attention/blob/main/DoWeNeedAttention.pdf)
* [CS 229S Systems for Machine Learning](https://cs229s.stanford.edu/fall2023/)
* [MLSys Seminar](https://mlsys.stanford.edu/)

**Table of contents:**
* [Foundation Models for Systems](#foundation-models-for-systems)
* [Hardware-Aware Algorithms](#hardware-aware-algorithms)
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
The canonical text book for everything FFT's: [Computational Frameworks for the Fast Fourier Transform](https://epubs.siam.org/doi/book/10.1137/1.9781611970999).

### Blog Posts
* [FlashAttention](https://crfm.stanford.edu/2023/01/13/flashattention.html)
* [FlashFFTConv](https://hazyresearch.stanford.edu/blog/2023-11-13-flashfftconv)

### Papers
* [FlashAttention](https://arxiv.org/abs/2205.14135) and [FlashAttention-2](https://arxiv.org/abs/2307.08691)
* [Self-Attention Does Not Need O(N^2) Memory](https://arxiv.org/abs/2112.05682)
* [FlashFFTConv: Efficient Convolutions for Long Sequences with Tensor Cores](https://arxiv.org/abs/2311.05908)
* [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
* [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889)

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
* [Diagonal State Spaces are as Effective as Structured State Spaces](https://arxiv.org/abs/2203.14343)
* [Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/abs/2307.08621)
* [Resurrecting Recurrent Neural Networks for Long Sequences](https://arxiv.org/abs/2303.06349)
* [MultiResFormer: Transformer with Adaptive Multi-Resolution Modeling for General Time Series Forecasting](https://arxiv.org/abs/2311.18780)
* [CKConv: Continuous Kernel Convolution For Sequential Data](https://arxiv.org/abs/2102.02611)

## Synthetics for Language Modeling
Synthetic tasks like associative recall have been very helpful in designing new architectures.

### Blog Posts
* MQAR blog post
* [H3 blog post](https://hazyresearch.stanford.edu/blog/2023-01-20-h3) section on associative recall

### Papers
* [H3 section 3.1](https://arxiv.org/abs/2212.14052)
* [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
* [Associative Long Short-Term Memory](https://arxiv.org/abs/1602.03032)
* [Using Fast Weights to Attend to the Recent Past](https://arxiv.org/abs/1610.06258)
* [Learning to update Auto-associative Memory in Recurrent Neural Networks for Improving Sequence Memorization](https://arxiv.org/abs/1709.06493)
* [Self-Attentive Associative Memory](https://arxiv.org/abs/2002.03519)

## Truly Sub-Quadratic Models

ML models are quadratic along another dimension -- model width.
Can we develop models that grow sub-quadratically with model width?

The canonical textbook for a lot of this stuff: [Structured Matrices and Polynomials](https://link.springer.com/book/10.1007/978-1-4612-0129-8).

### Blog Posts
* Monarch Blog Post
* [Monarch Mixer: Revisiting BERT, Without Attention or MLPs](https://arxiv.org/abs/2002.03519)
* M2 retrieval (hopefully)
* [Pixelated Butterfly: Simple and Efficient Sparse Training for Neural Network Models](https://hazyresearch.stanford.edu/blog/2022-01-17-Sparsity-3-Pixelated-Butterfly)
* [Butterflies Are All You Need: A Universal Building Block for Structured Linear Maps](https://dawn.cs.stanford.edu/2019/06/13/butterfly/)

### Papers
* [Monarch Mixer](https://arxiv.org/abs/2310.12109)
* [Monarch](https://arxiv.org/abs/2204.00595)
* [Pixelated Butterfly: Simple and Efficient Sparse training for Neural Network Models](https://arxiv.org/abs/2112.00029)
* [Learning Fast Algorithms for Linear Transforms Using Butterfly Factorizations](https://arxiv.org/abs/1903.05895)
* [Kaleidoscope: An Efficient, Learnable Representation For All Structured Linear Maps](https://arxiv.org/abs/2012.14966)
* [Fast Algorithms for Spherical Harmonic Expansions](http://tygert.com/butterfly.pdf)
* [Butterfly Factorization](https://arxiv.org/abs/1502.01379)
* [GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](https://arxiv.org/abs/2112.06905)
* [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635)
* [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149)
* [Learning to Prune Deep Neural Networks via Layer-wise Optimal Brain Surgeon](https://arxiv.org/abs/1705.07565)
* [A Two Pronged Progress in Structured Dense Matrix Multiplication](https://arxiv.org/abs/1611.01569)

## Systems for Inference
Inference is an increasingly important cost for LLMs: a model will be served many more times than it is trained.
Systems for inference are an increasingly important problem.
Here's some papers and posts on the topic, there's a lot to do!
* [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)
* [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
* [Flash-Decoding for long-context inference](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)
* [vLLM](https://github.com/vllm-project/vllm)
* [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
* [MatFormer: Nested Transformer for Elastic Inference](https://arxiv.org/abs/2310.07707)
* [Hugging Face TGI](https://huggingface.co/docs/text-generation-inference/index)
* [NVIDIA TensorRT](https://github.com/NVIDIA/TensorRT)
* [Together Inference Engine](https://www.together.ai/blog/together-inference-engine-v1)

## High-Throughput
Blog post https://hazyresearch.stanford.edu/blog/2023-04-12-batch

* FlexGen, Evaporate, blog post

## New Data Types
* HyenaDNA, find papers that have used it, blog posts, code, EEG, fMRI
