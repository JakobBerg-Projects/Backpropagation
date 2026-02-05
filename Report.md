# Report 
## Project by Jakob Berg & Tobias Munch

## Explaination of approach and design choices:

### Backpropagation

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


Overall approach

The backpropagation function implements layer-wise backpropagation by iterating backwards through the network, from the output layer 𝐿 to the first hidden layer. This directly follows the theoretical formulation of backpropagation, where gradients are computed using the chain rule, propagating error signals from later layers to earlier ones.

The implementation closely matches equations (4), (5), (6), and (7).

The for loop iterates backwards through each layer in the model. This reflects the definition of backpropagation, becouse the gradiant of each layer depends on the gradiant of the layer after it.

we have: delta = 2 * (y_pred - y_true) * model.df[i](model.z[i]), which relates to the equation for the output layer gradient (5). 

### Gradiant descent: 


