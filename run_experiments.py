from train_models import train_models
from save_recs_for_best_models import save_recs
from compute_metrics import compute_metric


######################## Train Models
train_models()

######################## Save RECS for Best Models
save_recs()

######################## Compute Metrics on Best Models
compute_metric()