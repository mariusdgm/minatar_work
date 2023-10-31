### 1. done, validated
Get a model, train for a bit with REDO enabled
check Q value for action before redo
Apply redo
check Q value for action after redo
they should be the same if we only did redo on dormant neurons

### 2.
we need to also reset optimizer at relevant indexes (reset momentum) 
pytorch adam holds 2 storages, one for momentum and one for hessian
exp_avgs and exp_avg_sqs get from adam.state maybe

### 3.
apply redo each 1000 training steps
use tau = 0.025, 0.1
use beta = 1 => makes the running avg actually be just the current value