# Building Blocks for AI Systems

This is a (biased) view of great work studying the building blocks of efficient and performant foundation models.
This Github was originally put together as a place to aggregate materials for a [NeurIPS keynote](https://neurips.cc/virtual/2023/invited-talk/73990) - but we're also hoping to highlight great work across AI Systems.
If you think we're missing something, please open an issue or PR!

Slides from Chris Ré's NeurIPS Keynote: https://cs.stanford.edu/~chrismre/papers/NeurIPS23_Chris_Re_Keynote_DELIVERED.pptx 

**Courses.** Courses a great resources for getting started in this space.
It's great that we have so many that have open materials!
Here's a partial list of courses -- it's biased by Stanford courses, so please reach out if you think of other resources that are helpful!
* [Stanford CS 324 LLMs](https://stanford-cs324.github.io/winter2022/)
* [Stanford CS 324 Advances in Foundation Models](https://stanford-cs324.github.io/winter2023/)
* [Sasha's talk on do we need attention?](https://github.com/srush/do-we-need-attention/blob/main/DoWeNeedAttention.pdf)
* [Stanford CS 229S Systems for Machine Learning](https://cs229s.stanford.edu/fall2023/)
* [MLSys Seminar](https://mlsys.stanford.edu/)
* [Berkeley AI-Sys](https://ucbrise.github.io/cs294-ai-sys-sp22/)
* [MIT CS 6.5940](https://hanlab.mit.edu/courses/2023-fall-65940)

If you just want to follow along on the major pieces from the talk, check out these blog posts:
* [Data Wrangling with Foundation Models](https://hazyresearch.stanford.edu/blog/2023-01-13-datawrangling)
* [FlashAttention](https://hazyresearch.stanford.edu/blog/2023-01-12-flashattention-long-sequences) and [FlashAttention-2](https://hazyresearch.stanford.edu/blog/2023-07-17-flash2)
* [Simplifying S4](https://hazyresearch.stanford.edu/blog/2022-06-11-simplifying-s4)
* [Long Convolutions for GPT-style Models](https://hazyresearch.stanford.edu/blog/2023-12-11-conv-tutorial)
* [Zoology Synthetics Analysis](https://hazyresearch.stanford.edu/blog/2023-12-11-zoology1-analysis)
* [Zoology Based](https://hazyresearch.stanford.edu/blog/2023-12-11-zoology2-based)
* [Truly Sub-Quadratic Models](https://hazyresearch.stanford.edu/blog/2023-12-11-truly-subquadratic)

An older set of resources on [Data-Centric AI](https://github.com/HazyResearch/data-centric-ai).

The rest of this README is split up into resources by topic.

**Table of contents:**
* [Foundation Models for Systems](#foundation-models-for-systems)
* [Hardware-Aware Algorithms](#hardware-aware-algorithms)
* [Can We Replace Attention?](#can-we-replace-attention)
* [Synthetics for Language Modeling](#synthetics-for-language-modeling)
* [Truly Sub-Quadratic Models](#truly-sub-quadratic-models)
* [Quantization, Pruning, and Distillation](#quantization-pruning-and-distillation)
* [Systems for Inference](#systems-for-inference)
* [High-Throughput](#high-throughput)
* [New Data Types](#new-data-types)

## Foundation Models for Systems
Foundation models are changing the ways that we build systems for classical problems like data cleaning.
[SIGMOD keynote](https://cs.stanford.edu/~chrismre/papers/SIGMOD-Chris-Re-DataCentric-Foundation-Models-KeyNote.pdf) on this topic.
Ihab Ilyas and Xu Chen's textbook on data cleaning: [Data Cleaning](https://dl.acm.org/doi/book/10.1145/3310205).
The [ML for Systems](https://mlforsystems.org/) workshops and community are great.

### Blog Posts
* [Bad Data Costs the U.S. $3 Trillion Per Year](https://hbr.org/2016/09/bad-data-costs-the-u-s-3-trillion-per-year)
* [Data Wrangling with Foundation Models](https://hazyresearch.stanford.edu/blog/2023-01-13-datawrangling)
* [Ask Me Anything: Leveraging Foundation Models for Private & Personalized Systems](https://hazyresearch.stanford.edu/blog/2023-04-18-personalization)

### Papers
* [Holoclean: Holistic Data Repairs with Probabilistic Inference](https://arxiv.org/abs/1702.00820)
* [Can Foundation Models Wrangle Your Data?](https://arxiv.org/abs/2205.09911)
* [Can Foundation Models Help Us Achieve Perfect Secrecy?](https://arxiv.org/abs/2205.13722) and [ConcurrentQA](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00580/117168/Reasoning-over-Public-and-Private-Data-in)
* [Language Models Enable Simple Systems for Generating Structured Views of Heterogeneous Data Lakes](https://arxiv.org/abs/2304.09433)
* [Symphony: Towards Natural Language Query Answering Over Multi-Modal Data Lakes](https://www.cidrdb.org/cidr2023/papers/p51-chen.pdf)
* [CodexDB: Synthesizing Code for Query Processing from Natural Language Instructions using GPT-3 Codex](https://www.vldb.org/pvldb/vol15/p2921-trummer.pdf)
* [CHORUS: Foundation Models for Unified Data Discovery and Exploration](https://arxiv.org/abs/2306.09610)
* [How Large Language Models Will Disrupt Data Management](https://dl.acm.org/doi/abs/10.14778/3611479.3611527)
* [GPTuner: A Manual-Reading Database Tuning System via GPT-Guided Bayesian Optimization](https://arxiv.org/abs/2311.03157)
* [Jellyfish: A Large Language Model for Data Preprocessing](https://arxiv.org/abs/2312.01678)
* [Can Large Language Models Predict Data Correlations from Column Names?](https://dl.acm.org/doi/abs/10.14778/3625054.3625066)

## Hardware-Aware Algorithms

Hardware-aware algorithms for today's ML primitives.
Canonical resources:
* A classic look at I/O complexity, from the database folks: [The input/output complexity of sorting and related problems](https://dl.acm.org/doi/10.1145/48529.48535).
* The canonical book on computer architectures: [Computer Architecture: A Quantitative Approach](https://ia800203.us.archive.org/31/items/2007ComputerArchitectureAQuantitativeApproach/2007%20-%20Computer%20Architecture%20-%20A%20Quantitative%20Approach.pdf).
* The canonical text book for everything FFT's: [Computational Frameworks for the Fast Fourier Transform](https://epubs.siam.org/doi/book/10.1137/1.9781611970999).

[Jim Gray's Turing Award Profile](https://amturing.acm.org/award_winners/gray_3649936.cfm).

### Blog Posts
* [Horace He's Making Deep Learning Go Brrrr from First Principles](https://horace.io/brrr_intro.html)
* [Aleksa Gordic's ELI5 for FlashAttention](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad)
* [FlashAttention](https://crfm.stanford.edu/2023/01/13/flashattention.html)
* [FlashFFTConv](https://hazyresearch.stanford.edu/blog/2023-11-13-flashfftconv)
* [Sasha's GPU Puzzles](https://github.com/srush/GPU-Puzzles)

### Papers
* [FlashAttention](https://arxiv.org/abs/2205.14135) and [FlashAttention-2](https://arxiv.org/abs/2307.08691)
* [Self-Attention Does Not Need O(N^2) Memory](https://arxiv.org/abs/2112.05682)
* [FlashFFTConv: Efficient Convolutions for Long Sequences with Tensor Cores](https://arxiv.org/abs/2311.05908)
* [tcFFT: Accelerating Half-Precision FFT through Tensor Cores](https://arxiv.org/abs/2104.11471)
* [Cooley-Tukey FFT Algorithm](https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm)
* [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
* [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889)
* [Faster Causal Attention Over Large Sequences Through Sparse Flash Attention](https://arxiv.org/abs/2306.01160)
* [FLAT: An Optimized Dataflow for Mitigating Attention Bottlenecks](https://arxiv.org/abs/2107.06419)
* [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
* [HOGWILD!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent](https://arxiv.org/abs/1106.5730)
* [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102)
* [Microsoft DeepSpeed](https://github.com/microsoft/DeepSpeed)
* [Eleuther's GPT-NeoX Repo](https://github.com/EleutherAI/gpt-neox)
* [A Systematic Approach to Blocking Convolutional Neural Networks](https://arxiv.org/abs/1606.04209)
* [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://arxiv.org/abs/1802.04799)
* [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://people.eecs.berkeley.edu/~matei/papers/2023/mlsys_megablocks.pdf)
* [Blockwise Self-Attention for Long Document Understanding](https://arxiv.org/abs/1911.02972)

## Can We Replace Attention?

Alternatives to attention that scale sub-quadratically in sequence length.
Canonical text on signal processing: [Discrete-Time Signal Processing](https://dl.acm.org/doi/10.5555/1795494).
High-level overview of this space: [From Deep to Long Learning](https://hazyresearch.stanford.edu/blog/2023-03-27-long-learning).

### Blog Posts
* [What is a long convolution?](https://hazyresearch.stanford.edu/blog/2023-12-11-conv-tutorial)
* [Can Longer Sequences Help Take the Next Leap in AI?](https://hazyresearch.stanford.edu/blog/2022-06-09-longer-sequences-next-leap-ai)
* [Simplifying S4](https://hazyresearch.stanford.edu/blog/2022-06-11-simplifying-s4)
* [Sasha's Great Annotated S4](https://srush.github.io/annotated-s4/)
* [H3: Language Modeling with State Space Models and (Almost) No Attention](https://hazyresearch.stanford.edu/blog/2023-01-20-h3)
* [Hyena Blog](https://hazyresearch.stanford.edu/blog/2023-06-08-hyena-safari)
* Mamba tweet threads by [Albert](https://twitter.com/_albertgu/status/1731727672286294400) and [Tri](https://twitter.com/tri_dao/status/1731728602230890895)
* [StripedHyena-7B](https://www.together.ai/blog/stripedhyena-7b)
* [Zoology](https://hazyresearch.stanford.edu/blog/2023-12-11-zoology0-intro)
* [Zoology Analysis](https://hazyresearch.stanford.edu/blog/2023-12-11-zoology1-analysis)
* [Based Architecture](https://hazyresearch.stanford.edu/blog/2023-12-11-zoology2-based)

### Papers
* [Long Range Arena](https://arxiv.org/abs/2011.04006)
* [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) and [code](https://github.com/state-spaces/mamba)
* [Zoology: Measuring and improving recall in efficient language models](https://arxiv.org/abs/2312.04927)
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
* [Pretraining Without Attention](https://arxiv.org/abs/2212.10544)
* [Diffusion Models Without Attention](https://arxiv.org/abs/2311.18257)
* [Liquid Structural State-Space Models](https://arxiv.org/abs/2209.12951)
* [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895)

### Attention Approximations
There's also a great literature around approximating attention (sparse, low-rank, etc).
Just as exciting!
Here's a partial list of great ideas in this area:
* [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236)
* [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451)
* [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794)
* [Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention](https://arxiv.org/abs/2102.03902)
* [Linformer: Self-Attention with Linear Complexity](https://arxiv.org/abs/2006.04768)
* [Skyformer: Remodel Self-Attention with Gaussian Kernel and Nyström Method](https://arxiv.org/abs/2111.00035)
* [Scatterbrain: Unifying Sparse and Low-rank Attention Approximation](https://arxiv.org/abs/2110.15343)
* [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062)
* [Luna: Linear Unified Nested Attention](https://arxiv.org/abs/2106.01540)
* [FNet: Mixing Tokens with Fourier Transforms](https://arxiv.org/abs/2105.03824)
* [The Devil in Linear Transformer](https://arxiv.org/abs/2210.10340)
* [cosFormer: Rethinking Softmax in Attention](https://arxiv.org/abs/2202.08791)

## Synthetics for Language Modeling
In research on efficient language models, synthetic tasks (_e.g._ associative recall) are crucial for understanding and debugging issues before scaling up to expensive pretraining runs.  

### Code
We've created a very simple GitHub repo with a simple playground for understanding and testing langauge model architectures on synthetic tasks: **[HazyResearch/zoology]( https://github.com/HazyResearch/zoology)**.

### Blog Posts
* [Zoology blog post on synthetics](https://hazyresearch.stanford.edu/blog/2023-12-11-zoology1-analysis)
* [H3 blog post](https://hazyresearch.stanford.edu/blog/2023-01-20-h3) section on associative recall
* [Anthropic's great explainer of associative recall in *induction heads*](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html#definition-of-induction-heads)

### Papers
* [Zoology section 3-4](https://arxiv.org/abs/2312.04927)
* [H3 section 3.1](https://arxiv.org/abs/2212.14052)
* [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
* [Associative Long Short-Term Memory](https://arxiv.org/abs/1602.03032)
* [Using Fast Weights to Attend to the Recent Past](https://arxiv.org/abs/1610.06258)
* [Learning to update Auto-associative Memory in Recurrent Neural Networks for Improving Sequence Memorization](https://arxiv.org/abs/1709.06493)
* [Self-Attentive Associative Memory](https://arxiv.org/abs/2002.03519)
* [Neural Turing Machines](https://arxiv.org/abs/1410.5401)
* [Legendre Memory Units: Continuous-Time Representation in Recurrent Neural Networks](https://papers.nips.cc/paper_files/paper/2019/hash/952285b9b7e7a1be5aa7849f32ffff05-Abstract.html)
* Synthetic tasks go all the way back to LSTMs: [Long Short-Term Memory](https://deeplearning.cs.cmu.edu/F23/document/readings/LSTM.pdf)

## Truly Sub-Quadratic Models

ML models are quadratic along another dimension -- model width.
Can we develop models that grow sub-quadratically with model width?

The canonical textbook for a lot of this stuff: [Structured Matrices and Polynomials](https://link.springer.com/book/10.1007/978-1-4612-0129-8).

### Blog Posts
* [Towards Truly Subquadratic Models](https://hazyresearch.stanford.edu/blog/2023-12-11-truly-subquadratic)
* [M2-BERT: Revisiting BERT, Without Attention or MLPs](https://hazyresearch.stanford.edu/blog/2023-07-25-m2-bert)
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
* [A Two Pronged Progress in Structured Dense Matrix Multiplication](https://arxiv.org/abs/1611.01569)

## Quantization, Pruning, and Distillation
Quantization, pruning, and distillation are great techniques to improve efficiency.
Here's just a short overview of some of the ideas here:
* [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
* [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635)
* [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149)
* [Learning to Prune Deep Neural Networks via Layer-wise Optimal Brain Surgeon](https://arxiv.org/abs/1705.07565)
* [QuIP#: QuIP with Lattice Codebooks](https://cornell-relaxml.github.io/quip-sharp/)
* [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438)
* [SpAtten: Efficient Sparse Attention Architecture with Cascade Token and Head Pruning](https://hanlab.mit.edu/projects/spatten)
* [Accelerating Inference with Sparsity Using the NVIDIA Ampere Architecture and NVIDIA TensorRT](https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/)
* [SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression](https://arxiv.org/abs/2306.03078)
* [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
* [MCUNet: Tiny Deep Learning on IoT Devices](https://arxiv.org/abs/2007.10319)
* [MONGOOSE: A Learnable LSH Framework for Efficient Neural Network Training](https://github.com/HazyResearch/mongoose)

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
* [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)
* [Hugging Face TGI](https://huggingface.co/docs/text-generation-inference/index)
* [NVIDIA TensorRT](https://github.com/NVIDIA/TensorRT)
* [Together Inference Engine](https://www.together.ai/blog/together-inference-engine-v1)
* [Laughing Hyena Distillery: Extracting Compact Recurrences From Convolutions](https://arxiv.org/abs/2310.18780)
* [H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://arxiv.org/abs/2306.14048)

## High-Throughput
Foundation models will increasingly be used to serve back-of-house tasks like document processing (not just chat interfaces).
These will require different systems than our current inference solutions.
This work is still very new, but hopefully there's a lot more to come soon!
* [Batch computing and the coming age of AI systems](https://hazyresearch.stanford.edu/blog/2023-04-12-batch).
* [FlexGen: High-throughput Generative Inference of Large Language Models with a Single GPU](https://arxiv.org/abs/2303.06865)
* [Evaporate: Language Models Enable Simple Systems for Generating Structured Views of Heterogeneous Data Lakes](https://www.vldb.org/pvldb/vol17/p92-arora.pdf)

## New Data Types
Most ML models focus on text or images, but there's a large variety of other modalities that present unique challenges (e.g., long context).
New modalities will drive advances in model architectures and systems.
A few modalities compiled below:
* DNA: [HyenaDNA paper](https://arxiv.org/abs/2306.15794) and [blog](https://hazyresearch.stanford.edu/blog/2023-06-29-hyena-dna)
* [SSMs for Video](https://arxiv.org/abs/2303.14526)
* [SpaceTime: Effectively Modeling Time Series with Simple Discrete State Spaces](https://arxiv.org/abs/2303.09489) [[paper](https://arxiv.org/abs/2303.09489)] [[code](https://github.com/HazyResearch/spacetime/tree/main)], [[demo](https://colab.research.google.com/drive/1dyR7ZGnjNfS2GMjRUfDzujQLhxSo-Xsk?usp=sharing)]
* [Recurrent Distance-Encoding Neural Networks for Graph Representation Learning](https://arxiv.org/abs/2312.01538)
* [Modeling Multivariate Biosignals With Graph Neural Networks and Structured State Space Models](https://arxiv.org/abs/2211.11176)
* [Self-Supervised Graph Neural Networks for Improved Electroencephalographic Seizure Analysis](https://arxiv.org/abs/2104.08336)
* [Self-Supervised Learning of Brain Dynamics from Broad Neuroimaging Data](https://arxiv.org/abs/2206.11417)
* [scHyena: Foundation Model for Full-Length Single-Cell RNA-Seq Analysis in Brain](https://arxiv.org/abs/2310.02713)
