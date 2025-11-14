# ML Paper Implementations

Working through influential machine learning papers chronologically to understand how the field evolved. The goal is to implement the core algorithms myself and see how ideas built on each other over time.

## What I'm Doing

I'm spending about 6 months implementing 1-2 papers per month, starting from the Perceptron in 1958 and working up to modern architectures like Transformers. For each paper, I'm:

- Building the core algorithm from scratch (when feasible)
- Testing it on appropriate datasets
- Documenting what I learned and what was challenging
- Creating visualizations to compare my results with the original paper

## Papers

### Foundations (1950s-1980s)
- [ ] Rosenblatt (1958) - The Perceptron
- [ ] Widrow & Hoff (1960) - ADALINE

### Neural Networks Revival (1980s-1990s)
- [ ] Rumelhart, Hinton & Williams (1986) - Backpropagation
- [ ] LeCun et al. (1989) - LeNet
- [ ] Hochreiter & Schmidhuber (1997) - LSTM

### Support Vector Machines (1990s)
- [ ] Cortes & Vapnik (1995) - Support Vector Networks
- [ ] Schölkopf et al. (1998) - Kernel PCA

### Ensemble Methods (1990s-2000s)
- [ ] Freund & Schapire (1997) - AdaBoost
- [ ] Breiman (2001) - Random Forests

### Deep Learning (2000s-2010s)
- [ ] Hinton et al. (2006) - Deep Belief Networks
- [ ] Krizhevsky et al. (2012) - AlexNet
- [ ] Goodfellow et al. (2014) - GANs
- [ ] He et al. (2015) - ResNet

### Modern Architectures (2015+)
- [ ] Vaswani et al. (2017) - Transformer
- [ ] Devlin et al. (2018) - BERT

## Setup

```bash
# Clone and set up environment
git clone https://github.com/yourusername/ml-paper-implementations.git
cd ml-paper-implementations
conda env create -f environment.yml
conda activate ml-papers
```

## Structure

Each paper gets its own folder in `papers/` with:
- Implementation code
- Jupyter notebooks for experiments
- README with paper summary and my notes
- Results and visualizations

The `common/` directory has shared utilities I use across multiple papers.

## Progress

| Paper | Status | Completed | Notes |
|-------|--------|-----------|-------|
| Perceptron (1958) | 📝 | - | Starting point |
| ADALINE (1960) | ⏸️ | - | |

Legend: 📝 In Progress | ✅ Done | ⏸️ Not Started