# Schema-tune: Enhancing Fairness in Transformer-based Language Models Through Subconscious-like Schema Updating and Cognitive Dissonance

# Requirements
The Main Requirements to run this project are `transformers==4.40.2` and `datasets==2.19.1`. A complete list of Requirements are provided in the `requirements.txt` file.
# Experiments on Bias Mitigation
- The code for performing Bias Mitigation through Cognitive Dissonance is stored in the files `LanguageModel.py`, `GaussianNoise.py` and `LearningAgent.py`
- The file `main2.py` contains the code for Bias Mitigation. It utilizes `LanguageModel.py`, `GaussianNoise.py` and `LearningAgent.py` and trains a GPT2 model.
- The parameters for all of our settings are stored in `Parameters/` folder.
- To train a model, run the `main2.py` file, and enter a number between 1 to 8 to choose the setting for the model you wish to train.
# Acknowledgements
This repository utilizes code from the following Github repository
- [An Empirical Analysis of Parameter-Efficient Methods for Debiasing Pre-Trained Language Models](https://github.com/x-zb/pedb?tab=readme-ov-file)
- [An Empirical Survey of the Effectiveness of Debiasing Techniques for Pre-trained Language Models](https://github.com/McGill-NLP/bias-bench)

# Citation


