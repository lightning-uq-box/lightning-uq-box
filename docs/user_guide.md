(guide)=

# Why lightning-uq-box

Over the past decade open-sourcing code alongside publications has luckily become a defacto standard. However, different papers continue to reimplemnt and develop methods in their own frameworks with a lot of boilerplate code. This can make it difficult to understand what the core aspects of a proposed method are. Additionally, this makes it more time-consuming to test various methods on your task of interests as each implementation might have different requirements and custom functionality. We aim to bridge this gap with the lightning-uq-box, which is essentially nothing more than a collection of [Lightning Modules](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html) that build on the shoulders of existing implementations and packages and give a common structure across methods.

With that comes the integration of [Lightning-CLI](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.cli.LightningCLI.html) which lets you run experiments at scale from the command line through configuration files to ensure reproducability. 

