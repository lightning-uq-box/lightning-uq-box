(guide)=

# User Guide

This document aims to give context and background around why we believe this project is beneficial to the community. Additionally, we want to explain the design principle of the approach we chose.


# Why lightning-uq-box

The initial motivation for developing the library was a curiosity about the various UQ-Methods that exist out there. It was quiet overwhelming in the beginning and it continues to be a fast moving field. We belive that a structured code implementation can often help with understanding a proposed method and also how it is similar or different than other approaches. 

Additionally, our background is the application side - specifically Earth Observation Data - where we are interested to what extend these methods can help our decision making tasks.

Over the past decade open-sourcing code alongside publications has luckily become a defacto standard. However, different papers continue to reimplemnt and develop methods in their own frameworks with a lot of boilerplate code. This can make it difficult to understand what the core aspects of a proposed method are. 

We are of course not the first to implement various methods in a more structured framework and in several instances we heavily rely on existing packages. The main contribution of the lightning-uq-box is the implementation of methods in the lightning framework which at its core allows users to structure their code in such a way that it removes unnecessary boilerplate code. This also has the advantage of giving a structured framwork around what exactly happens for a training or prediction step for example. Since all supported UQ-Methods are implemented as LightningModules, model training and evaluation can be done in the common lightning framework with its powerful Trainer class. Lightning additionally offers many utilities to ensure reproducibility. We harness that functionally through supporting LightningLI directly to both make it easier to run experiments at scale while focusing on reproducibility and minimizing code changes.

Overall we hope that the library can help researchers to more easily execute experiments with UQ Methods for their application task and to be able to quickly iterate through experiment ideas. Additionally, we also hope to help practicioners get into the field of UQ by providing extensive tutorials that explain basic concepts and illustrate these on toy problems so they can get an intuition for the supported methods.

- find a balance between giving enough flexibility but also structure to make methods widely applicable but with a common overarching structure

