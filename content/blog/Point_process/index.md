---
title: My Work -- Year 1 Individual Research Project
summary: Spatio-Temporal Analysis of Drug Crime Nearby, Using Kernel Density Estimation
date: 2024-10-21
authors:
  - Haochen
tags:
  - Project
---

All concepts mentioned in this blog are explained here: [Poster](/uploads/point_process.pdf)

In the last term of Year 1, each student in my class was required to complete an individual research project. It's only a single page poster. I was allocated the topic of point process. 

After studying several types of point processes, such as Possion Process, Renewal Process and so on, I think I want to find some method which is more general and easy to use, and of course, allow me to extend the topic. So I chose Kernel Density Estimation (KDE) as my method to analyze the data.

The whole project is easy to conclude. 

- Model the density distribution of KDE.
- Explain the relationship between density and intensity.
- Obtain the point process and analyze the final result.

Funny thing is that I implemented the KDE by cuml, which is a library designed for GPU accelerated machine learning. During my Year 1, I was looking forward to doing some machine learning projects, but I felt that I couldn't find a chance in a math department. After finish KDE on crime data, I realized that I had just finished a machine learning project.

Even though KDE doesn't have some important abilities for famous machine learning algorithms, it does follows that basic definition of machine learning which is algrithm that can learn from data and improve from experience. The more data I have, the more kernels I can put on the estimation, and the estimation will be closer to real distribution. KDE seems to be a good regression model for probability distribution.

This experience made me more believe in the idea that, machine learning is a branch of statistics. I have never seen articles, essays or books that define KDE as a machine learning algorithm, but it follows the definition. 