---
title: Study Notes - How I Understand Flow Matching, Starting from Generative Models.
summary: Explanation of flow-based generative model.
date: 2024-12-14
authors:
  - Haochen
tags:
  - Study Notes
  - Generative Models
image:
  caption:
---

Flow matching is one technique to train flow-based generative models. As a beginner in machine learning, I will explain my understanding of flow matching, starting from generative models.

# Generative Models

For a generative model, the key assumption is that, the sample we want to generate forms a distribution $p(x)$. For example, to train a model to generate images of cats, we want to learn a probability with dimension $256 \times 256 \times 3$. ($256 \times 256$ is the resolution of the image, and $3$ is the number of channels.) Samples from this distribution will be similar to cats.

$p(x)$ is a very complex distribution and in most cases, we don't know how it really looks like. One strategy is to learn a transformation from a simple distribution $p_Z(z)$ to $p_X(x)$, where $z$ is a simple distribution, like a Gaussian distribution. To generate samples, we can simple from $p_Z(z)$ and transform it to $p_X(x)$. This is the idea of flow-based generative models.

# Normalizing Flows

In the idea of Normalizing Flow, a learnable generator $G_{\theta}$ is built making:
$$
x = G_{\theta}(z)
$$
$$
p_{1}(x) = p_{0}(z) |det(J_{G^{-1}})|
$$
where $p_{0}(z)$ is the standard Gaussian distribution and $p_{1}(x)$ is the real distribution of data. 

$|det(J_{G^{-1}})|$ is the determinant of the Jacobian matrix of the inverse function of $G$, and its existence can be understood as the stretching of the probability space. Rigorous proof should be easily found from any statistics textbook.

The loss function of the model is defined by maximizing the log-likelihood, which is:
$$
log p_{1}(x) = log p_{0}(z) + log |det(J_{G^{-1}})|
$$

There are several restrictions on $G_{\theta}$. Firstly, it must be invertible. Secondly, $G_{\theta}$ must be in some special form to make the determinant easy to compute, such as triangular matrix. Due to the second restriction, single flow is not strong enough, so we need to make it deeper, which is:
$$
x = G_{\theta_{1}} \circ G_{\theta_{2}} \circ ... \circ G_{\theta_{n}}(z)
$$
and the log-likelihood is:
$$
log p_{1}(x) = log p_{0}(z) + \sum_{i=1}^{n} log |det(J_{G_{\theta_{i}}^{-1}})|
$$

# Continuous Normalizing Flows (CNFs)

There are several motivations for us to make the flow continuous. In my understanding, it makes the transformation more smooth and "continuous" can be understood as "infinitely deep". In CNFs, the transformation is defined by an ODE:
$$
\frac{d x_{t}}{d t} = u_t (x_t, \theta)
$$
for $t \in [0, 1]$. Here, $x_{t}$ is the state of the system at time $t$, and $u_t$ is the vector field. At time 0, $p_{0}(x)$ is the standard Gaussian distribution, and at time 1, $p_{1}(x)$ is the real distribution of data. Again, ignoring some calculus details, we have
$$
\frac{d p_{t}(x_t)}{d t} + \frac{d}{d x} (p_t(x_t)u_t(x_t)) = 0 
$$
$$
\frac{d p_{t}(x_t)}{d t} = - \nabla \cdot u_{t}(x_t, \theta) p_{t}(x_t)
$$
$$
\frac{d log p_{t}(x_t)}{d t} = - \nabla \cdot u_{t}(x_t, \theta)
$$
The Log-likelihood is:
$$
log p_{1}(x) = log p_{0}(x_0) + \int_{0}^{1} - \nabla \cdot u_{t}(x_t, \theta) dt
$$
In this equation, the biggest problem is that, the integral is very expensive to compute. This problem is exactly what flow matching is trying to solve.

# Flow Matching

By deriving loss function from likelihood, the model is directly learning the distribution of data. However, the distribution is fully defined by the transformation, which is flows. So that it's totally enough to learn the flows, which is the idea of flow matching. 

In flow matching, the goal is minimize the difference between neural network flows and the true flows, which is:
$$
L_{FM} = E_{t, p_t(x_t)} ||v_t(x_t, \theta) - u_t(x_t)||^2
$$
where $v_t(x_t)$ is the neural network flow and $u_t(x_t, \theta)$ is the true flow. However, the true flow is unknown. Solution is to use the conditional probability. 

Given a sample $z$, we obtain the condition $x_1 = z$. Under this condition, all the samples of $x_0$ flows to $z$. Under this conditional path, $u_t(x_t | x_1 = z)$ is a vector field all pointing to $z$. 

For the intermediate states, to gurantee $x_t$ starts from $x_0$ and ends at $x_1$, it's reasonable to define it as:
$$
x_t = \Psi_t(x_0 \mid x_1) = \textcolor{red}{\sigma_t} x_0 + \textcolor{blue}{\alpha_t} x_1
$$
where $\sigma_t + \alpha_t = 1$. 

Here we obtain the conditional flow matching:
$$
L_{CFM} = E_{t, q(z), p_t(x_t | x_1 = z)} ||v_t(x_t, \theta) - u_t(x_t|x_1 = z)||^2
$$
where $u_t(x_t|x_1 = z)$ is known.

Again, by some magic of calculus, we have:
$$
\nabla_{\theta} L_{FM} (\theta) = \nabla_{\theta} L_{CFM} (\theta)
$$
which is compeletely enough to do gradient descent and train the model.

# Flow Matching and Diffusion Models

To explain relationship between flow matching and diffusion models. I think it's necessary to explain the training algorithm of flow matching.

To train flow matching, we need to have the following samples.
$$
\begin{aligned}
    & t \sim \mathcal{U}([0, 1]) && \quad \text{\textcolor{green}{\(\triangleright\) sample time}} \\
    & z \sim q(z) && \quad \text{\textcolor{green}{\(\triangleright\) sample data}} \\
    & x_0 \sim p(x_0) && \quad \text{\textcolor{green}{\(\triangleright\) sample noise}} \\
    & x_t = \Psi_t(x_0 \mid x_1) && \quad \text{\textcolor{green}{\(\triangleright\) conditional flow}}
\end{aligned}
$$
And then gradient step with 
$$ \nabla_\theta \left\| v_t^\theta(x_t) - u_t(x_t|x_1 = z) \right\|^2
$$
One way to understand it that, given a sample from standard Gaussian distribution, we overlap it with a data point by function $\Psi_t(x_0|x_1)$ to get a $\text{\textcolor{red}{intermediate state}}$ $x_t$. And gradient step is to learn the flow to transform $x_t$ back to $z$.

Recall the diffusion model training, given a data point, we sample a noise from standard Gaussian distribution and overlap it with the data point to get a $\text{\textcolor{red}{noisy data point}}$. And then the neural network is trained to predict the noise.

Compare two statement, it's clear(maybe) that flow matching is indeed a diffusion model, where the denoising process is calculated and done by the flow.

# Reference
How I Understand Flow Matching : https://www.youtube.com/watch?v=DDq_pIfHqLs&t=873s

Flow Matching: Simplifying and Generalizing Diffusion Models | Yaron Lipman : https://www.youtube.com/watch?v=5ZSwYogAxYg

An Introduction to Flow Matching : https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html#flow-matching