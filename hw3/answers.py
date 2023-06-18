r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    hypers = dict(
        batch_size=256,
        seq_len=64,
        h_dim=512,
        n_layers=3,
        dropout=0.25,
        learn_rate=0.003,
        lr_sched_factor=0.3,
        lr_sched_patience=5,
    )
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    
    # ========================
    return start_seq, temperature


part1_q1 = r"""
We aim to divide the corpus while training for multiple reasons. By splitting the corpus into sequences, we can load manageable-sized examples into memory. This approach prevents the entire corpus from occupying all available memory. Another reason is we intend for the model to exclusively utilize recent data for predicting future characters. This allows us to minimize the impact of frequently occurring words from earlier in the corpus on our predictions.

"""

part1_q2 = r"""
The generation of text that shows longer memory than the sequence length is made possible by utilizing memory from many sources, including the previously generated text and the hidden state of the model. The final output is not restricted only by the maximum length of the preceding sequence. This ability allows the hidden state to hold more information than the sequence length, resulting in the generation of text that passes the maximum length.

"""

part1_q3 = r"""
Since each batch is trained with consideration of the of before, the order of the training batch matters. If we shuffle the training batch, the model will not be able to learn the relationship between the characters in the corpus as it is presented.

"""

part1_q4 = r"""
1. Lowering the temperature value affects the variability of the distribution for the next letter. By reducing the temperature to a value lower than 1, we can "highlight" some of the possible prediction options. This helps prevent the repetitive selection of the same letters throughout the prediction, resulting in a less deterministic model.

2. When the temperature is set to a high value, the distribution of outputs becomes highly variable. No single output stands out more than others, and the outputs are distributed uniformly.

3. When the temperature is set to a los value, we get adecrease in variability, resulting in a distribution with reduced variance. In this case, outputs with higher scores get higher chance to be predicted.


"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
   
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**


"""

part2_q2 = r"""
**Your answer:**


"""

part2_q3 = r"""
**Your answer:**



"""

part2_q4 = r"""
**Your answer:**


"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 0, 
        num_heads = 0,
        num_layers = 0,
        hidden_dim = 0,
        window_size = 0,
        droupout = 0.0,
        lr=0.0,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    
    # ========================
    return hypers




part3_q1 = r"""
**Your answer:**

"""

part3_q2 = r"""
**Your answer:**


"""


part4_q1 = r"""
**Your answer:**


"""

part4_q2 = r"""
**Your answer:**


"""


# ==============
