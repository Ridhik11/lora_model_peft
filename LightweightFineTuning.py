#UDACITY JUPYTER NOTEBOOK WORKSPACE - DOWNLOAD PYTHON FILE

# # Lightweight Fine-Tuning Project

# TODO: In this cell, describe your choices for each of the following
# 
# * PEFT technique: LoRA
# * Model: BERT
# * Evaluation approach: Hugging Face Trainer
# * Fine-tuning dataset: 'emotion'  

# ## Loading and Evaluating a Foundation Model
# 
# TODO: In the cells below, load your chosen pre-trained Hugging Face model and evaluate its performance prior to fine-tuning. This step includes loading an appropriate tokenizer and dataset.

# PROJECT GOAL - To analyze the performance of a pre-trained model and its fine-tuned version (both using the same BERT model) on a sequence classification task, following the fine-tuning of the pre-trained model with LoRA (PEFT).

# PROJECT FLOW:
# - Install and import the essetial libraries
# - Load the model (BERT), tokenizer associated with BERT, dataset (emotion)
# - Tokenize the dataset using tokenizer
# - Define the accuracy metric and compute metric function to evaluate the performance of the pre-trained BERT model on the dataset using Trainer.eval module.

# In[1]:


#Installing essential libraries
get_ipython().system('pip install transformers datasets peft evaluate scikit-learn')


# In[2]:


from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer


# In[3]:


from datasets import load_dataset
import evaluate #from sk-learn
import torch
import numpy as np 


# Importing the BERT model and tokenizer

# In[4]:


#Defining which model to load and then loading the model and associated tokenizer
model_bert = "bert-base-uncased"
#Pass 'model' into the Trainer and not 'model_bert'.
model = AutoModelForSequenceClassification.from_pretrained(model_bert, num_labels = 6) #Using num_labels = 6 because I am using 'emotion' dataset.
tokenizer = AutoTokenizer.from_pretrained(model_bert)


# Emotion is a dataset of English Twitter messages with six basic emotions: anger, fear, joy, love, sadness, and surprise. For more detailed information please refer to the paper.
# It has 20,000 rows and 6 labels.

# In[5]:


#Loading the dataset. 
dataset = load_dataset("emotion")


# In[6]:


#Tokenizing the laoded dataset
#Setting the padding= "max-length" and trucation = True so that each token is of same length. Padding adds and trucation cuts.
def tokenizer_func(examples):
    return tokenizer(examples["text"], padding = "max_length", truncation = True) 

tokenized_dataset = dataset.map(tokenizer_func, batched = True) 


# Till now we have completed the following: 
# - Installing and importing all essential libraries
# - Load the model, tokenizer and dataset. Tokenized the dataset using BERT associated tokenizer.
# 
# Next steps:
# - Deine accuracy metric using accuracy from evaluate module
# - Define the compute metric function to calculate the accuracy
# - Evaluate the pretrained model on the tokenized dataset using Tranier

# In[7]:


accuracy_metric = evaluate.load("accuracy") #The evaluate.load("accuracy") function internally uses sklearn.metrics.accuracy_score to compute the accuracy.


# During Training: Logits are PyTorch tensors (here predict)
# 
# During Evaluation: Logits get automatically converted to NumPy arrays before compute_metrics. Always assume evaluation predictions are NumPy arrays when using Hugging Face Trainer with PyTorch models.

# In[8]:


#Function to compute metric to evaluate the performance of the model. When passed inside the Trainer, it itself passes the prediction adn labels into the function as arguments (eval-prediction).
def compute(eval_prediction):
    predict, labels = eval_prediction
    prediction = np.argmax(predict, axis=-1)
    prediction = prediction.tolist()  # Convert to list
    labels = labels.tolist()  # Convert to list
    return accuracy_metric.compute(predictions = prediction, references = labels) #Accuracy metric that to be used


# In[9]:


#Initialize the trainer for evaluation of the model on test dataset. Need 3 things; model, dataset and compute metric on which  the is evaluated.
#We do not need to specify the argumetns for Trainer as it will itself pick up the default values.
#We only need to pass the eval_dataset to Trainer as we have evaluate the base model performance.
#We define 3 parameter for Trainer; model, dataset and compute metrics (function defined above).
trainer_base_eval = Trainer(
    model = model, #The model that needs to be evaluate
    eval_dataset = tokenized_dataset["test"], #The dataset on which the model is evaluated
    compute_metrics = compute
)


# In[10]:


#Evaluating the preformance/results for the abse model(BERT) using the trainer.evaluate()


# In[11]:


base_model_results = trainer_base_eval.evaluate()
print(f"Base model Results are as following:  {base_model_results}")


# In[12]:


# Print results in a formatted manner
print("Base Model Evaluation Results:")
print("---------------------------")
print(f"**Evaluation Loss:- {base_model_results['eval_loss']:.15f}")
print(f"**Evaluation Accuracy:- {base_model_results['eval_accuracy'] * 100:.2f}%")
print(f"**Evaluation Runtime:- {base_model_results['eval_runtime']:.2f} seconds")
print(f"**Samples per Second:- {base_model_results['eval_samples_per_second']:.2f}")
print(f"**Steps per Second:- {base_model_results['eval_steps_per_second']:.2f}")
print("---------------------------")


# ## Performing Parameter-Efficient Fine-Tuning
# 
# TODO: In the cells below, create a PEFT model from your loaded model, run a training loop, and save the PEFT model weights.

# In[13]:


from peft import LoraConfig, get_peft_model


# In[14]:


#Defining the configuration for lora model. 
lora_config = LoraConfig(
    r=8, #Rank of LoRA Matrix
    lora_alpha = 32, #Scaling factor
    target_modules = ['query', 'v_proj'], #Layer to apply LoRA to
    lora_dropout = 0.1, #Dropout rate
    bias = 'none',
    task_type = 'SEQ_CLS' #Sequence Classification
)


# In[15]:


# Save the adaptor_config file in the folder that will save all the essential files to load the fine tuned model.
lora_config.save_pretrained("fine_tune_lora_model") 


# In[16]:


lora_model = get_peft_model(model, lora_config) 
# Takes original pre-trained model (model) and injects LoRA adapters into the layers specified in lora_config.
# Only 0.1–5% of the model’s parameters are made trainable (the rest stay frozen).


# In[17]:


from transformers import TrainingArguments


# The Trainer class handles the entire training process, including iterating over the training data, computing losses, updating model weights, and evaluating performance

# In[18]:


#We do not need to specify the argumetns for Trainer as it will select the default values.
#This time we use Trainer to train (fine tune) the model (peft model) using .train()
#and we need to pass both training_dataset and eval_dataset to Trainer as we have to first train the model and then evaluate it performance.
#We pass define parameter for Trainer; model, dataset and compute metrics.
trainer_peft = Trainer(
    model = lora_model,
    train_dataset = tokenized_dataset['train'],
    eval_dataset = tokenized_dataset['test'],
    compute_metrics = compute
)


# In[20]:


#Fine tuning the base model. trainer.train(resume_from_checkpoint=True)
trainer_peft.train()


# In[26]:


# Save the fine-tuned model (adapter weights and PEFT configuration)
#I was doing a mistake here. I was saving the original base model instead of the lora_model that has been fine tuned.

lora_model.save_pretrained("fine_tune_lora_model")
tokenizer.save_pretrained("fine_tune_lora_model")


# ## Performing Inference with a PEFT Model
# 
# TODO: In the cells below, load the saved PEFT model weights and evaluate the performance of the trained PEFT model. Be sure to compare the results to the results from prior to fine-tuning.

# In[27]:


import os
print(os.listdir("fine_tune_lora_model"))


# In[31]:


print(lora_model.config.num_labels)


# In[32]:


# loading the fine-tuned model
from peft import AutoPeftModelForSequenceClassification

# Important to use this 'local_files_only=True' to define that the model is saved loacally and not on the hub. 
#Defining the num_lables for lora model to match the original base model.
fine_tune_lora_model = AutoPeftModelForSequenceClassification.from_pretrained("fine_tune_lora_model", num_labels = 6)
fine_tune_tokenizer = AutoTokenizer.from_pretrained("fine_tune_lora_model")


# In[33]:


#Using tokenizer of fne-tuned model to tokenize the test data.


# In[35]:


def tokenizer_func_fine_tune(examples):
    return fine_tune_tokenizer(examples["text"], padding = "max_length", truncation = True) 

fine_tune_dataset = dataset["test"].map(tokenizer_func_fine_tune, batched = True)


# In[36]:


#Using trainer modelu to define the evaluation condtion
trainer_fine_tune = Trainer(
    model= fine_tune_lora_model,
    eval_dataset = fine_tune_dataset ,
    compute_metrics = compute
)


# In[37]:


#Evaluation of the fine tuned model
fine_tuned_model_result = trainer_fine_tune.evaluate()
print(f"Fine-Tuned Model Evaluation Results: {fine_tuned_model_result}")


# In[38]:


# Print results in a formatted manner
print("Fine tune Model Evaluation Results:")
print("---------------------------")
print(f"**Evaluation Loss:- {fine_tuned_model_result['eval_loss']:.15f}")
print(f"**Evaluation Accuracy:- {fine_tuned_model_result['eval_accuracy'] * 100:.2f}%")
print(f"**Evaluation Runtime:- {fine_tuned_model_result['eval_runtime']:.2f} seconds")
print(f"**Samples per Second:- {fine_tuned_model_result['eval_samples_per_second']:.2f}")
print(f"**Steps per Second:- {fine_tuned_model_result['eval_steps_per_second']:.2f}")
print("---------------------------")


# In[40]:


get_ipython().system('pip install tabulate')


# In[41]:


from tabulate import tabulate

data = [
    ["Evaluation Loss", f"{base_model_results['eval_loss']:.15f}", f"{fine_tuned_model_result['eval_loss']:.15f}"],
    ["Evaluation Accuracy", f"{base_model_results['eval_accuracy'] * 100:.2f}%", f"{fine_tuned_model_result['eval_accuracy'] * 100:.2f}%"],
    ["Evaluation Runtime (s)", f"{base_model_results['eval_runtime']:.2f}", f"{fine_tuned_model_result['eval_runtime']:.2f}"],
    ["Samples per Second", f"{base_model_results['eval_samples_per_second']:.2f}", f"{fine_tuned_model_result['eval_samples_per_second']:.2f}"],
    ["Steps per Second", f"{base_model_results['eval_steps_per_second']:.2f}", f"{fine_tuned_model_result['eval_steps_per_second']:.2f}"],
]

print(tabulate(data, headers=["Metric", "Base Model", "Fine-Tuned Model"], tablefmt="fancy_grid"))


# The accuracy increased from 8.7% to 63.05% and decreased loss from 1.891 to 1.004. 
# I think the accuracy can be increased if I have used more 
# 'r' more than 8 and target each layer of the base model(target modules) and trained it for more number of epochs.
# 
# We currently have; r=8, #Rank of LoRA Matrix
#                    lora_alpha = 32, #Scaling factor
#                    target_modules = ['query', 'v_proj'], #Layer to apply LoRA to
