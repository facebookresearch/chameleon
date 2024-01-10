# Mixed-modal and Text-only Prompts for Human Evaluation

This file ```prompts_for_human_evaluations.jsonl``` contains the 1,048 prompts used for evaluating Chameleon's output: 441 (42.1%) are mixed-modal (i.e., containing both text and images), and the remaining 607 (57.9%) are text-only. The expected responses are mixed-modal, containing both text and images.

## Background

We work with a third-party crowdsourcing vendor to collect a set of diverse and natural prompts from human annotators. Specifically, we ask annotators to creatively think about what they want a multi-modal model to generate for different real-life scenarios. For example, for the scenario of “imagine you are in a kitchen”, annotators may come up with prompts like “How to cook pasta?” or “How should I design the layout of my island? Show me some examples.” The prompts can be text-only or text with some images, and the expected responses should be mixed-modal, containing both text and images.

After collecting an initial set of prompts, we ask three random annotators to evaluate whether the prompts are clear and whether they expect the responses to contain images. We use a majority vote to filter unclear prompts and prompts that don’t expect mixed-modal responses. In the end, our final evaluation set contains
1,048 prompts: 441 (42.1%) are mixed-modal (i.e., containing both text and images), and the remaining 607 (57.9%) are text-only.

More details on how these prompts are collected and some statistics can be found in the [paper](https://research.facebook.com/publications/).

## File format

Each line of the file ```prompts_for_human_evaluations.jsonl``` defines a prompt, with the following fields:
- ```id```: The GUID of this prompt.
- ```prompt```: The prompt content. If the prompt contains images, then their position is given by the special ```<img>``` token.
- ```task_type```: The task category of this prompt.
- ```image_urls```: A list of the URLs of images used in the prompts. Each image maps to a special ```<img>``` token in the prompt by order.