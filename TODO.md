### 1. done, actually not the same
Get a model, train for a bit with REDO enabled
check Q value for action before redo
Apply redo
check Q value for action after redo
they should be the same if we only did redo on dormant neurons

### 2. done
we need to also reset optimizer at relevant indexes (reset momentum) 
pytorch adam holds 2 storages, one for momentum and one for hessian
exp_avgs and exp_avg_sqs get from adam.state maybe

### 3. done
apply redo each 1000 training steps
use tau = 0.025, 0.1
use beta = 1 => makes the running avg actually be just the current value

### 4. 
use a varying tau to see the effect on max q diff

### 5. 
make histograms of redo scores, we must see if there is 
a significant diff between the scores between the conv and 
linear layers

### 6. 
maybe apply redo just in conv layers? don't touch the linear layer
we might need a different tau for the linear layer

### 7. done
unit test redo 
init small model
check redo reinit

## Comments
the scores are normalized in a layer so they sum to 1, but if we have enough neurons this means we increase the risk of all being dormant?