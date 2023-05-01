import pickle
import pandas as pd

myfile = "G:\\My Drive\\SWANN\\Dell for Pycharm Jan23 scientific\\outputs\\good_Duffing2model_switch2model_obviousdenoter"
filepath = 'G:\\My Drive\\SWANN\\Dell for Pycharm Jan23 scientific\\outputs\\good_Duffing2model_switch2model_obviousdenoter\\predict_by_ANN'

# objects = []
with (open(myfile+"\\predict_by_ANN", "rb")) as openfile:
    while True:
        try:
            # objects.append(pickle.load(openfile))
            all_pred_by_time = pickle.load(openfile)
        except EOFError:
            break

with (open(myfile+"\\bias_classification", "rb")) as openfile:
    while True:
        try:
            # objects.append(pickle.load(openfile))
            bias_classification = pickle.load(openfile)
        except EOFError:
            break

with (open(myfile + "\\bias_ODE", "rb")) as openfile:
    while True:
        try:
            # objects.append(pickle.load(openfile))
            bias_ODE = pickle.load(openfile)
        except EOFError:
            break

with (open(myfile + "\\softmax_vec", "rb")) as openfile:
    while True:
        try:
            # objects.append(pickle.load(openfile))
            softmax_vec = pickle.load(openfile)
        except EOFError:
            break

with (open(myfile + "\\wts_ODE", "rb")) as openfile:
    while True:
        try:
            # objects.append(pickle.load(openfile))
            wts_ODE = pickle.load(openfile)
        except EOFError:
            break


with (open(myfile + "\\wts_classification", "rb")) as openfile:
    while True:
        try:
            # objects.append(pickle.load(openfile))
            wts_classification = pickle.load(openfile)
        except EOFError:
            break

with (open(myfile + "\\predict_by_weights", "rb")) as openfile:
    while True:
        try:
            # objects.append(pickle.load(openfile))
            predict_by_wts = pickle.load(openfile)
        except EOFError:
            break
# fig4name = "\\dynamics_measured_vs_ANN_4coeffs_2switches1.pickle"
# with (open(myfile+fig4name, "rb")) as openfile:
#     while True:
#         try:
#             # objects.append(pickle.load(openfile))
#             fig4 = pickle.load(openfile)
#         except EOFError:
#             break

# obj = pd.read_pickle(r'filepath')

x=1