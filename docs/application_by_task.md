(application_by_task)=

# Application of UQ-Methods by Task

There exist a wide variety of UQ-Methods. Similarly, there are several general tasks for which practicioners might require uncertainty estimates. The library currently supports the following four tasks:

1. Regression for tabular/image inputs with 1D scalar targets
2. 2D Regression / Pixel Wise Regression
3. Classification for tabular/image inputs with single classification label
4. Segmentation where each pixel is assigned a class

While some UQ-Methods like MC-Dropout or Deep Ensembles can be applied across tasks, other methods are specifically developed for certain tasks. The following aims to give an overview of supported methods for the different tasks.

## Regression 1D targets

```{toctree}
:maxdepth: 1

tutorials/regression1D
```

## Pixelwise Regression

## Classification with single labels

## Segmentation