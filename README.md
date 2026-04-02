This implementation was inspired by two pioneering works in multi-modal graph-based recommendation:

- **MGAT** ([github.com/zltao/MGAT](https://github.com/zltao/MGAT)) - Multi-modal Graph Attention Network.
- **MMGCN** ([paper](https://liqiangnie.github.io/paper/MMGCN.pdf)) - Multi-modal Graph Convolution Network for micro-video recommendation.
  We adopted its core principle of building separate graphs per modality.

## Overview

MGAT is a graph-based recommender system that learns user preferences by combining **text**, **entity**, and **category** information from news articles.

## Data

- **MIND dataset** - Microsoft News Recommendation dataset
- User interactions (clicks) + article features (text embeddings, entity embeddings, category IDs)
- Graph structure: users + items as nodes, interactions as edges

## Architecture

1. **Multi-modal features** → Three parallel GNNs process text, entity, and category embeddings
2. **Graph Attention (GAT)** → Nodes attend to neighbors to learn representations
3. **Fusion** → Modality outputs averaged for final user/item embeddings
4. **Loss** → Pairwise ranking with Bayesian Personalized Ranking (BPR)

## Evaluation (Top-K)

Following MIND competition standards:

- **Metrics**: Precision@10, Recall@10, NDCG@10
- **How it works**: For each user, rank all candidate items by embedding similarity, select top-10, compare against held-out clicked articles

## Output

Top-10 news recommendations ranked by relevance score.
