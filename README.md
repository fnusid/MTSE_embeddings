This repo contains model and training code for extracting multiple speaker emebddings from noisy mixture. This doesn't require number of speakers information prior, but instead computes speaker emebdding recursively until a stop criterion.

## Installing requirements
`pip install -r requirements.txt`

## Changing parameters
You can change the model, dataset, loss and the training parameters in `config.py`

## Training code
You can start training the model by running the following in the terminal
`python trainer.py`
