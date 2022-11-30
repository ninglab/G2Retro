import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

BOND_SIZE = 3
MAX_NB = 6

VALENCE_NUM = 8
SUB_CHARGE_NUM = 3
SUB_CHARGE_CHANGE_NUM = 3

HYDROGEN_NUM = 6
IS_RING_NUM = 2
IS_CONJU_NUM = 2
IS_AROMATIC_NUM = 2
REACTION_CLS = 10

SUB_CHARGE_OFFSET = 1
