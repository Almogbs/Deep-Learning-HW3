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

    hypers['batch_size'] = 512
    hypers['seq_len'] = 64
    hypers['h_dim'] = 128
    hypers['n_layers'] = 3
    hypers['dropout'] = 0.1
    hypers['learn_rate'] = 1e-3
    hypers['lr_sched_factor'] = 1e-1
    hypers['lr_sched_patience'] = 5

    return hypers


def part1_generation_params():
    start_seq = "Yo William man, how u doin? "
    temperature = 0.5

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
    hypers["betas"] = (0.9, 0.999)
    hypers["batch_size"] = 32
    hypers["h_dim"] = 256
    hypers["z_dim"] = 128
    hypers["x_sigma2"] = 5
    hypers["learn_rate"] = 0.0005

    return hypers


part2_q1 = r"""
Your answer:
When sigma is set to a high value, the loss gives more importance to the kldiv loss. This gives a smoother representation of the data. As a result, the generated samples become more similar to each other.
When sigma is set to a small value, the loss depends more on the difference between the original and reconstructed image. This means the model will try to keep the details of the input image as accurate as possible.
Thus, sigma is responsible for regulating the VAE model, when a large-> lowers the generated images diversity, adn a smaller sigma-> many different looking images but less organized or meaningful
"""

part2_q2 = r"""
Your answer:
1. 
The reconstruction loss aims to generate images that closely resemble the original input images. On the other hand, the KL divergence loss aims to check that the two images that are close to each other in the latent space will result in two similar reconstructed images. This means a smooth latent space.


2.
The KL loss term has a direct influence on the variance of the latent space. When the KL loss is larger, it leads to a higher variance in the latent space. This means that different latent vectors can be more spread out and have a wider range of values.
On the other hand, a smaller KL loss results in a reduced variance of the latent space-> the latent vectors are more concentrated and have less variability in their values.

3.
We can achive better generalization by ensuring that similar input images result in similar reconstructed outputs. This allows the model to capture and preserve important features or patterns present in the data, leading to more accurate reconstructions.
By smoothly transitioning between different regions of the latent space, the model can generate meaningful and diverse samples.

"""

part2_q3 = r"""
Your answer:
The VAE model aims to generate images that closely resemble real data. However, the space of possible options for an image of size 3x64x64 is incredibly big. By maximizing the evidence distribution, P(X), and by that the likelihood, we prioritize generating images that are closer to the dataset distribution. This will increase the chances of producing high-quality, realistic images that resemble the characteristics of the training data.
"""

part2_q4 = r"""
Your answer:
we model the log of the latent-space variance corresponding to an input, instead of directly modelling this variance because the need for the variance to be a positive value. By using the log, we make sure that the resulting variance remains positive when applying the exponential function. 
"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 256, 
        num_heads = 4,
        num_layers = 8,
        hidden_dim = 64,
        window_size = 8,
        droupout = 0.1,
        lr=0.0001,
    )

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