import torch
from transformers import BertConfig, BertForPreTraining, load_tf_weights_in_bert

def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):
    # Initialize the model from the BERT configuration
    config = BertConfig.from_json_file(bert_config_file)
    print("Building PyTorch model from configuration: {}".format(config))
    model = BertForPreTraining(config)
    
    # Load weights from tf checkpoint
    load_tf_weights_in_bert(model, config, tf_checkpoint_path)
    
    # Save the PyTorch model
    print("Saving PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)

# Specify paths
tf_checkpoint_path = '../models/bert-base-uncased/bert_model.ckpt'  # path to the TensorFlow checkpoint (excluding the data-00000-of-00001 suffix)
bert_config_file = '../models/bert-base-uncased/config.json'  # path to the config file
pytorch_dump_path = '../models/pytorch_bert_checkpoint/pytorch_model.bin'  # path to save the new PyTorch model

# Call the conversion function
convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path)