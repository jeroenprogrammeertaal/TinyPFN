import math
import torch
import numpy as np

from tinygrad.tensor import Tensor

from data.preprocess import prepare_inputs

class TinyMLP:

  def __init__(self, input_dim, hidden_dim, out_dim, dropout_p=0.5):
    self.dropout_p = dropout_p
    self.l1 = Tensor.scaled_uniform(input_dim, hidden_dim)
    self.b1 = Tensor.zeros(hidden_dim)
    self.l2 = Tensor.scaled_uniform(hidden_dim, out_dim)
    self.b2 = Tensor.zeros(out_dim)

  def forward(self, x: Tensor):
    return x.matmul(self.l1).add(self.b1).gelu().dropout(self.dropout_p).matmul(self.l2).add(self.b2)
  
  def get_parameters(self):
    return [self.l1, self.l2, self.b1, self.b2]

class TransformerBlock:

  def __init__(self, hidden_dim: int, head_dim: int, n_heads: int, ff_size: int, dropout_p):   
    self.head_dim = head_dim
    self.n_heads = n_heads
    self.dropout_p = dropout_p
    self.hidden_dim = hidden_dim
    self.projection = Tensor.scaled_uniform(hidden_dim, 3 * head_dim * n_heads)
    self.projection_b = Tensor.zeros(3 * head_dim * n_heads)
    self.o_projection = Tensor.scaled_uniform(hidden_dim, hidden_dim)
    self.o_projection_b = Tensor.zeros(hidden_dim)
    self.ln1_weight = Tensor.ones(hidden_dim)
    self.ln1_bias = Tensor.zeros(hidden_dim)
    self.ln2_weight = Tensor.ones(hidden_dim)
    self.ln2_bias = Tensor.zeros(hidden_dim)
    self.mlp = TinyMLP(hidden_dim, ff_size, hidden_dim, dropout_p=dropout_p)

  def attn(self, x: Tensor, eval_pos):
    # Shape: Batch (B) * Dataset/Sequence (S) * hidden_dim (D) x 3
    B, _, D = x.shape
    D = D // 3

    q = x[:, :, :self.hidden_dim]
    k = x[:, :, self.hidden_dim : self.hidden_dim * 2]
    v = x[:, :, self.hidden_dim * 2:]

    def attn(q: Tensor, k: Tensor, v: Tensor):
        q = q.reshape(B, q.shape[1], self.n_heads, D // self.n_heads).transpose(1, 2)
        k = k.reshape(B, k.shape[1], self.n_heads, D // self.n_heads).transpose(1, 2)
        v = v.reshape(B, v.shape[1], self.n_heads, D // self.n_heads).transpose(1, 2)

        y = q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1])
        y = y.softmax(-1) @ v

        y = y.transpose(1, 2).reshape(B, q.shape[2], D)
        return y


    q_train, k_train, v_train = q[:, :eval_pos], k[:, :eval_pos], v[:, :eval_pos]
    q_test = q[:, eval_pos:]

    attn_left = attn(q_train, k_train, v_train)
    attn_right = attn(q_test, k_train, v_train)
    return attn_left.cat(attn_right, dim=1)

  def forward(self, x: Tensor, n_training_examples: int):
    # x Shape: B * S * D
    # projected Shape: B * S * 3 x D
    # return shape -> B * S * D

    projected = x.matmul(self.projection).add(self.projection_b)
    projected = self.attn(projected, n_training_examples).dropout(self.dropout_p)
    projected = projected.matmul(self.o_projection).add(self.o_projection_b)

    x = x + projected.dropout(self.dropout_p)
    x = x.layernorm().mul(self.ln1_weight).add(self.ln1_bias)

    x2 = self.mlp.forward(x)
    x = x + x2.dropout(self.dropout_p)
    return x.layernorm().mul(self.ln2_weight).add(self.ln2_bias)

  def get_parameters(self):
    params = [self.projection, self.projection_b,self.o_projection, self.ln1_weight, self.ln1_bias, self.ln2_weight, self.ln2_bias]
    params.extend(self.mlp.get_parameters())
    return params

class TinyPFNTransformer:

  def __init__(self, vocab_size: int, embedding_size: int, ff_size: int, n_layers: int, dropout_p: float):
    self.x_embedding = Tensor.glorot_uniform(vocab_size, embedding_size)
    self.x_embedding_b = Tensor.zeros(embedding_size)
    self.y_embedding = Tensor.glorot_uniform(1, embedding_size)
    self.y_embedding_b = Tensor.zeros(embedding_size)
    self.layers = [TransformerBlock(512, 128, 4, ff_size, dropout_p) for _ in range(n_layers)]
    self.decoder = TinyMLP(embedding_size, ff_size, 10, 0.0)

  def forward(self, x: Tensor, y: Tensor, n_training_examples: int):
    n_test_examples = y.shape[1] - n_training_examples
    y_test_embedding = Tensor.zeros(y.shape[0], n_test_examples, self.y_embedding.shape[-1], requires_grad=False)
    embedded = x.linear(self.x_embedding, self.x_embedding_b)
    embedded_y = y[:, :n_training_examples].linear(self.y_embedding, self.y_embedding_b).cat(y_test_embedding, dim=1)
    embedded = embedded.add(embedded_y)
    for layer in self.layers:
      embedded = layer.forward(embedded, n_training_examples) 
    return self.decoder.forward(embedded)
  
  def get_class_probs(self, outputs: Tensor, configs):
    agg_logits = Tensor.zeros_like(outputs[0])
    for i, config in enumerate(configs):
      (_, class_shift), _, _ = config
      output_ = outputs[i, :]
      if class_shift > 0:
        agg_logits += output_[..., class_shift:].cat(output_[..., :class_shift], dim=-1)
      
    agg_logits /= len(configs)
    probs = agg_logits.softmax(-1)
    return probs
  
  def get_parameters(self):
    params = [self.x_embedding, self.y_embedding]
    for layer in self.layers:
      params.extend(layer.get_parameters())
    params.extend(self.decoder.get_parameters())
    return params
  
  def get_n_parameters(self):
    params = self.get_parameters()
    return sum([p.numel() for p in params])

def load_ckpt_weights(tiny_model):

  ckpt_params = torch.load('model_ckpt/model.cpkt', map_location=torch.device('cpu'))[0]
  for n, p in ckpt_params.items():
    if 'module.transformer_encoder.layers.' in n:
      name = n.replace('module.transformer_encoder.layers.', '')
      n_split = name.split('.')
      layer = int(n_split[0])
      module = n_split[1]
      part = n_split[2]

      if module == 'norm1':
        if part == 'weight':
          tiny_model.layers[layer].ln1_weight = Tensor(p.numpy()).reshape(tiny_model.layers[layer].ln1_weight.shape)
        elif part == 'bias':
          tiny_model.layers[layer].ln1_bias = Tensor(p.numpy()).reshape(tiny_model.layers[layer].ln1_bias.shape)
      elif module == 'norm2':
        if part == 'weight':
          tiny_model.layers[layer].ln2_weight = Tensor(p.numpy()).reshape(tiny_model.layers[layer].ln2_weight.shape)
        elif part == 'bias':
          tiny_model.layers[layer].ln2_bias = Tensor(p.numpy()).reshape(tiny_model.layers[layer].ln2_bias.shape)

      elif module == 'self_attn':
        if part == 'in_proj_weight':
          tiny_model.layers[layer].projection = Tensor(p.T.numpy())
        elif part == 'in_proj_bias':
          tiny_model.layers[layer].projection_b = Tensor(p.numpy()).reshape(tiny_model.layers[layer].projection_b.shape)
        elif part == 'out_proj':
          if n_split[-1] == 'weight':
            tiny_model.layers[layer].o_projection = Tensor(p.T.numpy())
          elif n_split[-1] == 'bias':
            tiny_model.layers[layer].o_projection_b = Tensor(p.numpy()).reshape(tiny_model.layers[layer].o_projection_b.shape)
      
      elif module == 'linear1':
        if part == 'weight':
          tiny_model.layers[layer].mlp.l1 = Tensor(p.T.numpy())
        elif part == 'bias':
          tiny_model.layers[layer].mlp.b1 = Tensor(p.numpy()).reshape(tiny_model.layers[layer].mlp.b1.shape)
      elif module == 'linear2':
        if part == 'weight':
          tiny_model.layers[layer].mlp.l2 = Tensor(p.T.numpy())
        elif part == 'bias':
          tiny_model.layers[layer].mlp.b2 = Tensor(p.numpy()).reshape(tiny_model.layers[layer].mlp.b2.shape)

    elif 'module.y_encoder' in n:
      if 'weight' in n:
        tiny_model.y_embedding = Tensor(p.numpy()).reshape(tiny_model.y_embedding.shape)
      elif 'bias' in n:
        tiny_model.y_embedding_b = Tensor(p.numpy()).reshape(tiny_model.y_embedding_b.shape)
    elif 'encoder' in n:
      if 'weight' in n:
        tiny_model.x_embedding = Tensor(p.T.numpy())
      elif 'bias' in n:
        tiny_model.x_embedding_b = Tensor(p.numpy()).reshape(tiny_model.x_embedding_b.shape)
    
    elif 'decoder.0.weight' in n:
      tiny_model.decoder.l1 = Tensor(p.T.numpy())
    elif 'decoder.0.bias' in n:
      tiny_model.decoder.b1 = Tensor(p.numpy()).reshape(tiny_model.decoder.b1.shape)
    elif 'decoder.2.weight' in n:
      tiny_model.decoder.l2 = Tensor(p.T.numpy())
    elif 'decoder.2.bias' in n:
      tiny_model.decoder.b2 = Tensor(p.numpy()).reshape(tiny_model.decoder.b2.shape)


if __name__ == "__main__":
  model = TinyPFNTransformer(100, 512, 1024, 12, 0.5)
  load_ckpt_weights(model)

  dummy_x = np.random.rand(24, 25)
  dummy_y = np.random.randint(0, 9, (24, 1))

  x, y, configs = prepare_inputs(dummy_x, dummy_y, 32)
  x = Tensor(x)
  y = Tensor(y)

  output = model.forward(x, y, 12)
  probs = model.get_class_probs(output, configs)
  print(probs.numpy())