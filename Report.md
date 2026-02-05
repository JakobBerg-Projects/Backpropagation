<h1 align="center">Report</h1> 
<h3 align="center">Project by Jakob Berg & Tobias Munch</h3> 


<h2 align="center">Explaination of approach and design choices:</h2> 

## Backpropagation:


### Overall approach:

The backpropagation function implements layer-wise backpropagation by iterating backwards through the network, from the output layer 𝐿 to the first hidden layer. This directly follows the theoretical formulation of backpropagation, where gradients are computed using the chain rule, propagating error signals from later layers to earlier ones.

The implementation closely matches equations (4), (5), (6), and (7).

The for loop iterates backwards through each layer in the model. This reflects the definition of backpropagation, becouse the gradiant of each layer depends on the gradiant of the layer after it.


\[
\frac{\partial L}{\partial w^{[l]}_{i,j}}
= \delta^{[l]}_i \, a^{[l-1]}_j
\quad \forall l \in [1..L]
\] 

\[
\delta^{[l]}_i
= \frac{\partial L}{\partial z^{[l]}_i}
= \frac{\partial L}{\partial a^{[l]}_i}
\cdot
\frac{\partial a^{[l]}_i}{\partial z^{[l]}_i}
  (4)
\] 



\[
\delta^{[L]}_i
= \frac{\partial L}{\partial \hat{y}_i}
\cdot
\frac{\partial \hat{y}_i}{\partial z^{[L]}_i}
= e'_i(\hat{y}_i) \cdot f'^{[L]}_i(z^{[L]}_i)
\] 

\[
e'_i(y_i, \hat{y}_i) = -2 (y_i - \hat{y}_i)
(5)
\] 



\[
\delta^{[l]}_i
= \left(
\sum_{k=1}^{n^{[l+1]}}
\delta^{[l+1]}_k w^{[l+1]}_{k,i}
\right)
\cdot
f'^{[l]}_i(z^{[l]}_i)
(6)
\] 



\[
\frac{\partial L}{\partial b^{[l]}_j}
= \delta^{[l]}_j
(7)
\] 

### Output layer gradients:
we have: `delta = 2 * (y_pred - y_true) * model.df[i](model.z[i])`, which relates to the equation for the output layer gradient (5).  `2*(y_pred  - y_true)` comes from the derivative of the MSE loss \[
e'_i(y_i, \hat{y}_i) = -2 (y_i - \hat{y}_i)
\]  (the sign is absorbed in the subtraction order). 
`model.df[i](model.z[i])` represents: \[f'^{[l]}_i(z^{[l]}_i)\], which is the derivative of the activation function in the input layer.
Thuss delta stores the local gradient. 

### The hidden layer gradients:
`W_next = model.fc[str(i+1)].weight`
`delta = (delta @ W_next) * model.df[i](model.z[i])`, 
comes from equation (6). 
`delta @ W_next` computes the weighted sums of downstream errors and `model.df[i](model.z[i])` is like before, the derivative of the activationfunction.

### The weight gradients: 
`A_prev = model.a[i-1]`
`model.dL_dw[i] = (delta.t() @ A_prev)`
This code relates to eqation (4). `(delta.t() @ A_prev)` computes all the weights in one matrix operation. This avoids nested loops and follows the standard outer-product formulations. Using stored activations like `a[i-1]` avoids recomputation. 

### The bias gradients:
`model.dL_db[i] = delta.sum(dim=0)`, 
is an direct implementation of equation (7). We get the graients by summing over the deltas over the batch. This reflect the fact that the bias inputs are constant. The summation accounts for batch-wise gradient accumulation. 

### Conclusion:
This implementation faithfully follows the theoretical backpropagation equations (4–7). The use of vectorized operations ensures computational efficiency while preserving a clear correspondence to the mathematical definitions of local gradients, weight updates, and bias updates.


### Gradiant descent: 


