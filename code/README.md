# Σχεδιασμός και Αξιολόγηση Συστημάτων Ομοσπονδιακής Μάθησης για την Παραγωγή Προτάσεων με βάση γράφους

Design and Evaluation of Federated Learning Systems for Graph-Based Recommendation

## Requirements
- Python 3.10+
- GPU με CUDA (αργή εκπαίδευση χωρίς)
- Πρόσβαση στο dataset `wyzelabs/RuleRecommendation` στο [HuggingFace](https://huggingface.co/datasets/wyzelabs/RuleRecommendation)

## Εγκατάσταση
```bash
cd code
python -m venv .venv
source .venv/bin/activate
pip install -e .
```
(τα dependencies βρίσκονται στο `pyproject.toml` και εγκαθίστανται αυτόματα)

## Dataset
Τα δεδομένα κατεβαίνουν αυτόματα από το HuggingFace κατά την πρώτη εκτέλεση και αποθηκεύονται στον φάκελο `data/`. Αν απαιτείται ταυτοποίηση:

```bash
huggingface-cli login
# ή
export HF_TOKEN = <your_token>
```

## Εκτέλεση runs
Η εκπαίδευση εκτελείται με το Flower, επιλέγοντας ένα από τα τέσσερα σενάρια ενορχήστρωσης:

```bash
flwr run . <federation>
```

| Federation | Σενάριο | Clients |
|---|---|---|
| `centralized` | Centralized | 1 |
| `cross-silo` | Cross-silo | 4 |
| `pseudo-cross-device` | Pseudo cross-device | 2000 |
| `cross-device` | True cross-device | 25000 |

Οι υπερπαράμετροι εκπαίδευσης ορίζονται στο `[tool.flwr.app.config]` του `pyproject.toml` και μπορούν να γίνουν override απευθείας:

```bash
flwr run . cross-silo --run-config "num-rounds=50 lr=5e-4"
```
## Απαραίτητες παράμετροι ανά federation
Ανάλογα με το ποια federation αρχιτεκτονική εκτελείται, αλλάζουν οι παράμετροι των clients:

| Federation | `min-available-clients` | `min-fit-clients` | Επιπλέον παράμετροι |
|---|---|---|---|
| `centralized` | 1 | 1 | - |
| `cross-silo` | 4 | 4 | - |
| `pseudo-cross-device` | 2000 | 2000 | - |
| `cross-device` | 25000 | 15000 | `fraction-fit=0.6`, `max-train-users=25000` |

## Επιλογή έκδοσης μοντέλου
Η έκδοση επιλέγεται μέσω των παραμέτρων `model`, `encoder`, `decoder` στο `pyproject.toml`:

| Έκδοση | `model` | `encoder` | `decoder` | Περιγραφή |
|---|---|---|---|---|
| v1 | `simple` | — | — | GraphSAGE + MLP, χωρίς τύπους ακμών |
| v2 | `edge` | `sage` | `mlp` | pseudo-embeddings + GraphSAGE + MLP |
| v3 | `edge` | `compgcn` | `complex` | embeddings + CompGCN + ComplEx |

Παράδειγμα για v2 centralized run:
```
flwr run . centralized --run-config 'model="edge" encoder="sage" decoder="mlp" min-available-clients=1 min-fit-clients=1'
```
## Αξιολόγηση
Μετά την εκπαίδευση, το καλύτερο μοντέλο αποθηκεύεται ως `results/run_<timestamp>/best_model.pt`. Για την τελική αξιολόγηση χρησιμοποιούνται δύο εντολές, ανάλογα με το πρωτόκολλο:

Full leave-one-out (cold-start στο test set):
```bash
python helpers/final_eval.py --model results/run_<timestamp>/best_model.pt
```

Second place (single edge, MRR@50):
```bash
python helpers/final_eval.py --model results/run_<timestamp>/best_model.pt --second-place
```

## Δομή του κώδικα
| Αρχείο | Περιεχόμενο |
|---|---|
| `dataset.py` | Φόρτωση/καθαρισμός δεδομένων, λεξιλόγιο, μετατροπή σε γράφους |
| `model.py` | Ορισμός των μοντέλων V1/V2/V3 |
| `task.py` | Leave-one-out, negative sampling, τοπική εκπαίδευση & αξιολόγηση |
| `client_app.py` | Ορισμός των Flower clients |
| `server_app.py` | Ορισμός του Flower server |
| `results_logger.py` | Καταγραφή μετρικών, checkpoints, resume |
| `pyproject.toml` | Dependencies, υπερπαράμετροι, ορισμός των federations, configuration |
| `helpers/` | Βοηθητικά scripts (τελική αξιολόγηση, tests, audit) |