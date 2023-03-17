SETTING_MAIN_OUTPUT_DIR = "../data/training_runs"
SETTING_DATA_PRODUCTS_SUBDIR = "data_products"
SETTING_PLOTS_SUBDIR = "plots"

SETTING_INPUTS_FILE = "classifier.csv"  # also contains run outcomes
SETTING_LINE_FILE = "regressor.csv"     # also contains inputs
SETTING_CONTINUUM_FILE = "tba.csv"

SETTING_FORCE_TRAINING_STOP_ENABLED = True

SETTING_NORMALISE_INPUTS = True         # normalise all inputs to [0,1]
SETTING_TRANSFORM_LINE_DATA = True      # e.g. take log10 of (emission line data + 1)
SETTING_TRANSFORM_LINE_DATA_OFFSET = 1.0e-50

SETTING_TRAIN_FRACTION = 0.8
SETTING_VAL_FRACTION = 0.1
SETTING_TEST_FRACTION = 0.1

SETTING_NUM_EPOCHS = 100
SETTING_BATCH_SIZE = 32
SETTING_LEARNING_RATE = 0.001
SETTING_LEARNING_RATE_GAMMA = 0.99
SETTING_LEARNING_RATE_STEP_SIZE = 1
SETTING_P_DROPOUT = 0.2
SETTING_EVAL_FREQ = 1
SETTING_TEST_FREQ = 100

DEEPNUBLADO_INPUTS = [
    "gas_density",
    "gas_phase_metallicity",
    "redshift",
    "cr_ionization_factor",
    "ionization_parameter",
    "stellar_metallicity",
    "stellar_age",
    "dtm",
]
DEEPNUBLADO_CLASSIFIER_OUTPUTS = ["status"]
DEEPNUBLADO_REGRESSOR_OUTPUTS = [
    "C_2_157.636m",
    "O_3_88.3323m",
    "O_3_51.8004m",
    "O_4_25.8832m",
    "O_1_63.1679m",
    "O_1_145.495m",
    "N_3_57.3238m",
    "N_2_205.244m",
    "N_2_121.767m",
    "C_4_1550.78A",
    "C_4_1548.19A",
    "H_1_6562.80A",
    "H_1_4861.32A",
    "C_3_1908.73A",
    "O_2_3728.81A",
    "O_2_3726.03A",
    "O_3_1666.15A",
    "N_2_6583.45A",
    "N_5_1238.82A",
    "N_5_1242.80A",
    "HE_2_1640.41A",
    "O_3_5006.84A",
    "O_3_4958.91A",
    "S_2_6730.82A",
    "S_2_6716.44A",
    "NE_3_3868.76A",
    "CO_2600.05m",
    "CO_1300.05m",
    "CO_866.727m",
    "CO_650.074m",
    "CO_520.089m",
    "CO_433.438m",
    "CO_371.549m",
    "^13CO_2719.67m",
    "^13CO_1359.86m",
    "^13CO_906.599m",
    "^13CO_679.978m",
    "HCO+_3360.43m",
    "HCO+_1680.21m",
    "HCO+_1120.18m",
    "HCO+_840.150m",
    "HCO+_672.144m",
    "HCO+_560.140m",
    "HCN_3381.44m",
    "HCN_3381.58m",
    "HCN_3381.52m",
    "HCN_1690.82m",
    "HCN_1690.78m",
    "HCN_1690.80m",
    "HCN_1127.22m",
    "HCN_845.428m",
    "HCN_676.373m",
    "HCN_563.665m",
    "HCN_483.168m",
    "HCN_422.796m",
    "HCN_375.844m",
]

SCALER_MIN_INPUTS = {
    "gas_density": -2.999831499071751,
    "gas_phase_metallicity": 0.0010003612892792,
    "redshift": 3.0003293149450787,
    "cr_ionization_factor": 10.001837505823708,
    "ionization_parameter": -3.999633603071072,
    "stellar_metallicity": -4.999971911629007,
    "stellar_age": 1005804.0060347028,
    "dtm": 2.164899139652425e-05,
}

SCALER_MAX_INPUTS = {
    "gas_density": 5.999708474196375,
    "gas_phase_metallicity": 1.9971400289754475,
    "redshift": 11.998954062769748,
    "cr_ionization_factor": 999.7675259685866,
    "ionization_parameter": -0.0001360442426179,
    "stellar_metallicity": -1.397922145860092,
    "stellar_age": 1999340515.1910343,
    "dtm": 0.4999914124103983,
}
