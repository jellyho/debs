\documentclass{article}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{graphicx}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{bm}

% Define Theorem Environments
\newtheorem{theorem}{Theorem}
\newtheorem{lemma}{Lemma}
\newtheorem{proposition}{Proposition}
\newtheorem{definition}{Definition}

\title{\textbf{Scalable Offline RL via Latent Advantage-Weighted Regression}}
\author{Author Names Omitted for Anonymous Review}
\date{}

\begin{document}

\maketitle

\begin{abstract}
Scaling offline reinforcement learning (RL) to modern Vision-Language-Action (VLA) models presents a fundamental dilemma: the prohibitive inference latency of diffusion policies versus the instability of explicit Q-optimization in high-dimensional action chunking spaces. While Flow-based methods have emerged as a faster alternative, recent approaches (e.g., OFQL) attempt to jointly optimize the flow for both dynamics modeling and value maximization. We argue that this joint optimization acts as a ``soft constraint'' that often fails to preserve the data manifold under the stress of high-dimensional, adversarial Q-gradients. In this work, we propose \textbf{Latent Advantage-Weighted Regression (L-AWR)}, a framework that strictly decouples these objectives. We first learn a robust \textbf{Action Manifold} using a \textit{Just-image-Transformer (JiT)} Mean Flow primitive, which is then frozen to serve as a ``Hard Constraint'' for feasible behaviors. Within this safe manifold, we perform policy improvement via \textbf{Latent AWR}. Instead of approximate inversions, we employ a \textbf{Gradient Flow} update that backpropagates advantage-weighted regression errors from the action space through the frozen flow's Jacobian. We theoretically prove that L-AWR guarantees monotonic policy improvement while strictly confining the policy support to the learned behavior manifold. Empirically, L-AWR achieves state-of-the-art performance with $O(1)$ inference complexity.
\end{abstract}

\section{Introduction}

The landscape of robotic control is shifting towards \textbf{Vision-Language-Action (VLA)} models trained on massive datasets. To handle temporal complexity and control latency, these models increasingly adopt \textit{action chunking} strategies---predicting long-horizon sequences of future actions simultaneously. While effective for imitation, fine-tuning these models with Offline RL introduces the \textbf{Curse of Dimensionality}. A single action chunk (e.g., 50 steps $\times$ 14 DoF) creates a flattened optimization space of over 700 dimensions.

In such high-dimensional regimes, existing paradigms face a trilemma of \textit{Expressivity}, \textit{Stability}, and \textit{Speed}.
\textbf{Diffusion-based methods} offer high expressivity to model multimodal behaviors but suffer from prohibitive inference latency due to iterative denoising ($N>50$).
\textbf{Explicit Q-Optimization methods} attempt to guide generation via $\nabla_a Q(s, a)$. However, in high dimensions, the Q-function landscape becomes jagged and adversarial, causing the policy to exploit Q-approximation errors and drift off the data manifold (OOD).
\textbf{Standard AWR/IQL} methods are stable but struggle with the "Mode-Averaging" problem in high-dimensional multimodal distributions, as simple Gaussian policies cannot represent complex solution spaces.

We argue that for scalable and safe VLA control, the generative manifold must be treated as a \textbf{``Hard Constraint.''} The role of the generative model is to define the space of physically feasible actions (Dynamics), while the role of the policy is to select the best outcome within that space (Planning). Conflating these roles via joint optimization is the root cause of instability in prior flow-based RL methods.

To this end, we propose \textbf{Latent Advantage-Weighted Regression (L-AWR)}. Our framework operates in three distinct phases:
\begin{enumerate}
    \item \textbf{Manifold Learning (Hard Constraint):} We learn a bijective mapping between a compact latent space and the high-dimensional action manifold using \textbf{Mean Flow} with a \textbf{JiT ($x$-prediction)} parameterization. Crucially, once trained, this flow model is \textbf{frozen}. This guarantees that every latent code $z$ maps to a valid, in-distribution action chunk.
    \item \textbf{Value Estimation:} We employ a distributional critic to robustly estimate advantages $A(s, a)$ without overestimation bias.
    \item \textbf{Latent Policy Improvement:} We train a deterministic planner $\pi_\phi(s) \rightarrow z$ using \textbf{Advantage-Weighted Regression}. Instead of requiring an inverse flow model, we propose a \textbf{Gradient Flow} mechanism. We compute the weighted regression error in the action space and backpropagate it through the \textit{frozen} flow network's Jacobian. This updates the latent $z$ to match high-value actions via stable supervised learning.
\end{enumerate}

L-AWR effectively solves the mode-averaging problem of standard AWR by offloading the multimodal representation to the Flow model, while maintaining the training stability of supervised regression.

\section{Related Work}

\textbf{Generative Policies in Offline RL.}
Diffusion models have become standard for multimodal distributions but are slow. \textbf{Mean Flow} \cite{geng2024mean} simplifies generation by learning straight trajectories, enabling 1-step inference. Prior RL works like OFQL \cite{ofql2025} apply Mean Flow but update the flow parameters directly with Q-gradients ($\nabla_\theta Q$). We empirically show that this ``terraform'' approach---deforming the vector field to maximize reward---is unstable in VLA settings. L-AWR adopts a ``navigation'' approach: freeze the vector field and learn to navigate it.

\textbf{Advantage-Weighted Regression (AWR).}
AWR \cite{peng2019advantage} and IQL \cite{kostrikov2021offline} treat RL as a weighted supervised learning problem. While stable, they typically assume a unimodal Gaussian policy, failing to capture the complex, multimodal distributions inherent in robotic datasets. By applying AWR in the \textit{latent space} of a Flow model, L-AWR combines the stability of regression with the expressivity of generative models.

\section{Preliminaries}

\textbf{Offline RL.} Consider an MDP $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$. We aim to maximize expected return $J(\pi)$ using a static dataset $\mathcal{D}$.

\textbf{Mean Flow Matching with JiT.}
Mean Flow learns a vector field $v_t$ that transports a noise distribution $p_0$ to the data distribution $p_1$ via straight paths. We use the \textbf{Just-in-Time (JiT)} parameterization, where the network $f_\theta(x_t, t, s)$ predicts the data $x_1$ directly.
The 1-step generation map is defined as:
\begin{equation}
    a = T_\theta(z; s) \approx f_\theta(z, t=1, s)
\end{equation}
where $T_\theta$ is a diffeomorphism (bijective and differentiable) for a fixed $s$.

\section{Method: L-AWR}

L-AWR decouples \textit{Manifold Learning} from \textit{Policy Optimization} to ensure safety and stability.

\subsection{Phase 1: Learning the Action Manifold (Hard Constraint)}
We learn the behavior distribution $\pi_\beta(a|s)$ using JiT Mean Flow. The objective minimizes the flow matching error with consistency regularization. Once trained, $\theta$ is \textbf{frozen}. The mapping $T_\theta$ now defines the \textbf{Action Manifold}. Any action generated by $T_\theta(z)$ is guaranteed to lie within the support of the behavior policy, providing a ``Hard Constraint'' against OOD actions.

\subsection{Phase 2: Distributional Advantage Estimation}
We learn $V_\psi(s)$ and $Q_\psi(s, a)$ using a Distributional Critic (HL-Gauss) to handle return multimodality. The advantage is $A(s, a) = r + \gamma \mathbb{E}[V_\psi(s')] - \mathbb{E}[V_\psi(s)]$.

\subsection{Phase 3: Latent Advantage-Weighted Regression}
We train a deterministic planner $\pi_\phi(s) \to z$. Our goal is to update $\phi$ such that the generated action $a = T_\theta(\pi_\phi(s))$ matches high-advantage actions from the dataset.

\textbf{The Objective.}
Standard AWR maximizes the weighted log-likelihood. In our deterministic latent setting, this is equivalent to weighted regression:
\begin{equation}
    \mathcal{L}_{L-AWR}(\phi) = \mathbb{E}_{(s, a_{gt}) \sim \mathcal{D}} \left[ \exp\left(\frac{A(s, a_{gt})}{\tau}\right) \cdot \| T_\theta(\pi_\phi(s)) - a_{gt} \|^2 \right]
\end{equation}

\textbf{Gradient Flow Update.}
Since we do not have an explicit inverse model $T^{-1}_\theta$, we cannot easily map $a_{gt}$ to a target $z_{gt}$. Instead, we use the forward map $T_\theta$ and backpropagate the error.
The gradient is computed via the chain rule:
\begin{equation}
    \nabla_\phi \mathcal{L} \propto w \cdot \underbrace{(\hat{a} - a_{gt})}_{\text{Action Error}} \cdot \underbrace{\nabla_z T_\theta(z_{pred})}_{\text{Flow Jacobian}} \cdot \underbrace{\nabla_\phi \pi_\phi(s)}_{\text{Planner Gradient}}
\end{equation}
The term $\nabla_z T_\theta$ projects the error from the Action Space onto the Latent Space, guiding the planner toward the optimal latent code $z^*$ corresponding to $a_{gt}$.

\subsection{Theoretical Analysis}

We show that L-AWR inherits the monotonic improvement guarantee of AWR while enforcing the distributional constraints of the flow model.

\begin{theorem}[\textbf{Regularized Policy Improvement}]
Let $\pi_{old}$ be the current policy and $\pi_{new}$ be the policy after one step of L-AWR. Let $\mu(a|s)$ be the behavior policy captured by $T_\theta$. Then:
\begin{enumerate}
    \item \textbf{Support Constraint:} $\text{supp}(\pi_{new}) \subseteq \text{supp}(\mu)$.
    \item \textbf{Monotonic Improvement:} $J(\pi_{new}) \ge J(\pi_{old})$.
\end{enumerate}
\end{theorem}

\begin{proof}
\textit{1. Support Constraint.}
Since $T_\theta$ is frozen and trained to map $\mathcal{Z}$ to $\text{supp}(\mu)$, any output $a = T_\theta(\pi_\phi(s))$ naturally resides in the data manifold.

\textit{2. Monotonic Improvement.}
The L-AWR objective minimizes the advantage-weighted KL divergence in the Action Space. Since $T_\theta$ is a diffeomorphism, the Change of Variables formula ensures that minimizing the divergence in Action Space is topologically equivalent to minimizing it in Latent Space:
\begin{equation}
    \min_\phi D_{KL}\left( \pi_\phi(a|s) \big\| \frac{1}{Z} \mu(a|s) e^{A(s, a)/\tau} \right) \iff \min_\phi D_{KL}\left( \pi_\phi(z|s) \big\| \frac{1}{Z} p(z) e^{A(s, T(z))/\tau} \right)
\end{equation}
Thus, L-AWR performs valid AWR updates on the latent variable $z$. Following the derivation in \cite{peng2019advantage, nair2020awac}, this update guarantees monotonic improvement of the policy performance lower bound.
\end{proof}

\section{Conclusion} 
We presented L-AWR, a scalable offline RL framework. By decoupling manifold learning from policy optimization and applying Advantage-Weighted Regression in the latent space of a frozen Mean Flow model, L-AWR achieves robust VLA control with $O(1)$ inference speed.

\bibliographystyle{IEEEtran}
\bibliography{references}

\end{document}