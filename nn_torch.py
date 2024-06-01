import torch

# Softmax function
def softmax(logits):
    exp_logits = [logit.exp() for logit in logits]
    sum_exp_logits = sum(exp_logit for exp_logit in exp_logits)
    out = [exp_logit / sum_exp_logits for exp_logit in exp_logits]
    return out

# Negative log likelihood loss function
def negative_log_likelihood(logits, label):
    probs = softmax(logits)
    return -torch.log(probs[label])

# Example usage
logits = [torch.tensor(0.0, requires_grad=True),
          torch.tensor(3.0, requires_grad=True),
          torch.tensor(-2.0, requires_grad=True),
          torch.tensor(1.0, requires_grad=True)]

loss = negative_log_likelihood(logits, 3)  # dim 3 acts as the label for this input example
loss.backward()

# Print loss and gradients
print(f"Negative Log Likelihood Loss: {loss.item()}")

torch_answer=[0,0,0,0]
ans = [0.041772570515350445, 0.8390245074625319, 0.005653302662216329, -0.8864503806400986]
for dim in range(4):
    ok = 'OK' if abs(logits[dim].grad.item() - ans[dim]) < 1e-5 else 'WRONG!'
    torch_answer[dim]=logits[dim].grad.item()
    print(f"{ok} for dim {dim}: expected {ans[dim]}, yours returns {logits[dim].grad.item()}")

def torch_answer(x):
    answer=[0,0,0,0]
    for i in range(4):
        answer[i]=logits[i].grad.item()
    return answer[x]
