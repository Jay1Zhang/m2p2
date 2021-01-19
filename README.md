# M2P2



## M2P2 Framework



training procedure

```tex
Input:	train_dataset, val_dataset, N, n
Output:	Multi-modality models, weights

Initialize three unimodal reference models

% Master Procedure Start
for epoch = 1, ..., N do
	Train mod_model, Get latent embeddings $H^latent_m$ with $H^in_m$ in Equation(1)(2)
	Train shared_mlp_model, Get alignment embeddings $H^align$ and $H^s_m$ with $H^latent_m$ in Equation(3)(5)
	Calculate $L^align$ with $H^s_m$ in Equation(4)
	
	% Slave Procedure Start
	for epoch=1, ...,n do
		Train ref_mlp_model
		Eval ref_mlp_model, Calculate $L^ref_m$ with $H^latent_m$ in Equation(6)(7)
		Update modality weights $w_m$ with $L^ref_m$ in Equation(9)(10)
	% Slave procedure End
	
	Get $H^het$ with $w_m$ and $H^latent_m$ in Equation(8)
	
	Calculate $L^pers$ with $H^align$, $H^het$ and $X_meta$ in Equation(11)
	Calculate $L^final$ with $L^pers$ and $L^align$ in Equation(12)
	Update all parameters.
	
% Master Procedure End
```



