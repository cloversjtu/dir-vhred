# Dirichlet Latent Variable Hierarchical Recurrent Encoder-Decoder in dialogue generation（EMNLP2019）


# Description
This repository hosts the Dir-VHRED model for dialogue generation as described by Min and Yisen et al.2019


# requirements
pandas==0.20.3
numpy==1.14.0
gensim==3.1.0
spacy==1.9.0
tqdm==4.15.0
nltk==3.2.5
tensorboardX==1.1
torch==0.4


# 1.Prepare the dataset:
1.download the ubuntu corpus in http://dataset.cs.mcgill.ca/ubuntu-corpus-1.0/
& movie corpus in https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
2.split the dataset with train/validation/test rate: 0.8，0.1，0.1
3.run code:
 python3 cornell_preprocess.py
 python3 ubuntu_preprocess.py



# 2.Train the model:

python3 train.py --data=ubuntu  --batch_size=40 --eval_batch_size=40  --kl_annealing_iter=100000 --word_drop=0.25 --z_sent_size=3

python3 train.py --data=cornell  --batch_size=40 --eval_batch_size=40  --kl_annealing_iter=20000 --word_drop=0.25 --z_sent_size=3




# 3.Evaluate the model(negative log-likelihood):

python3 eval.py --data=ubuntu   --batch_size=40 --eval_batch_size=40   --z_sent_size=3  --checkpoint=xxxx

python3 eval.py --data=cornell  --batch_size=40 --eval_batch_size=40  --z_sent_size=3   --checkpoint=xxxx


# 4.Evaluate the model(word-embedding metric):
python3 eval_embed.py --data=ubuntu   --batch_size=40 --eval_batch_size=40   --z_sent_size=3   --checkpoint=xxxx --beam_size=5  --n_sample_step=3
#n_sample_step=1 means 1-turn while n_sample_step=3 means 3-turn responses

python3 eval_embed.py --data=cornell   --batch_size=40 --eval_batch_size=40   --z_sent_size=3  --checkpoint=xxxx --beam_size=5 --n_sample-step=3

# 5.Generate the response:
python3 generate_sentence.py --data=ubuntu   --batch_size=40  --kl_annealing_iter=100000 --word_drop=0.25 --z_sent_size=3 --checkpoint=xxx

python3 generate_sentence.py --data=cornell   --batch_size=40  --kl_annealing_iter=20000 --word_drop=0.25 --z_sent_size=3 --checkpoint=xxx
