# Embedding Model Training Approach

In this approach, the goal is to train the embedding model so that tokens of similar classes are brought closer together while tokens of different classes are pushed apart.

To achieve this, we will leverage a pre-trained language model (PLM) as well as two auxiliary neural networks called projection networks denoted by $f_\mu$ and $f_\Sigma$. These neural networks will generate the means and covariances of token distributions. The fundamental idea here is that tokens follow a normal distribution.

Given a text of size $n$: $[x_1, \dots, x_n]$ representing the tokens, we will extract the embedding representations of each of these tokens $[\textbf{h}_1, \dots, \textbf{h}_n]$ ($h_j \in \mathbb{R}^d$ where $d$ depends on the choice of PLM).

$$
[\textbf{h}_1, \dots, \textbf{h}_n] = PLM([x_1, \dots, x_n])
$$

We will then use the projection networks to construct the parameters of a Gaussian distribution:

$$
{\mu}_i = f_\mu({h}_i), \ (\Sigma_i)_k = ELU(f_\Sigma(h_i)_k) + (1 + \epsilon)
$$

We assume that $h_i$ follows a normal distribution $\mathcal{N}(\mu_i, \Sigma_i)$. In this distribution, $\Sigma_i$ is a diagonal matrix whose entries are determined by the equation above. The ELU (Exponential Linear Unit) is an activation function that provides a differentiable variation of the ReLU (Rectified Linear Unit) activation function:

$$
ELU(x) = 
\begin{cases}
    x & \text{if } x \geq 0, \\
    \alpha (e^x - 1) & \text{if } x < 0
\end{cases}
$$

Here, $\alpha$ is a hyperparameter and typically, we use $\alpha = 1$. We add $(1 + \epsilon)$ to $\Sigma_i$ to ensure that it is positive definite, which is a necessary condition for a covariance matrix.

We define the distance between two normal distributions $\mathcal{N}_p$ and $\mathcal{N}_q$ as the Kullback-Leibler divergence (KL), which measures the dissimilarity between two probability distributions in terms of information gain:

$$
\begin{split}
    d(p,q) &= \frac{1}{2}\left[ D_{KL}(\mathcal{N}_p || \mathcal{N}_q) + D_{KL}(\mathcal{N}_q || \mathcal{N}_p) \right]
\end{split}
$$

Ultimately, given a training point $p$, we define $X_p \subset X$ as the set of points $(x,y)$ belonging to the same class as $p$. Thus, our objective is to minimize the softmax sum of distances between $X_p$ and all other points in $X$:

$$
\begin{split}
    l(p) & = -\log{\left[
        \frac{\sum_{(X_q, y_q) \in X_p}\exp{(-d(p,q))} /|X_p|}{\sum_{X_q, y_q \in X, p \neq q} \exp{(-d(p,q))}}
        \right]
    } \\
    \mathcal{L}(X)  &= \frac{1}{|X|} \sum_{p \in X} l(p)
\end{split}
$$

This approach aims to reduce the distance between two tokens belonging to the same class, making this distance tend towards $0$.

Figure \ref{fig:container} shows a training schema of ContaiNER. The author also proposes a two-stage training, given a Source training set (with plenty of available observations) and our few-shot learning set called Support. First, we will perform training of the PLM, $f_\mu$, and $f_\Sigma$ on the Source set (which can take a long time), then we will fine-tune on the Support set.

After these two trainings, inference is performed using a nearest neighbor method seen during training (k-NN): given a test sentence $[x_1, \dots, x_m]$, label predictions are given by $[\hat{y}_1, \dots, \hat{y}_m]$.

$$
\begin{cases}
    [h_1, \dots h_m] = PLM(x_1, \dots, x_m) \\
    \hat{y}_i = kNN(h_i)
\end{cases}
$$

An advantage of ContaiNER over other few-shot learning approaches is that, since there is no text transformation before passing through the language model, ContaiNER becomes the only approach where LayoutLM (the state-of-the-art for document image classification task) is applicable, thus allowing the use of 2D positional embeddings.

However, the ContaiNER approach, which involves simultaneously optimizing three neural networks, requires considerable computational power. The training step requires approximately 20 GB of graphics memory, and the entire process can take over an hour, even with very little data. Additionally, implementing this method is not simple, and managing the numerous hyperparameters poses a major challenge.
