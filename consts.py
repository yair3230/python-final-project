# Occupation mapping on humane-realistic scale (0-100)
ISCO_MAPPING = {
    '11': 85,
    '12': 75,
    '21': 95,
    '22': 55,
    '23': 35,
    '24': 70,
    '26': 15,
    '32': 65,
    '33': 60,
    '34': 50,
    '41': 45,
    '42': 30,
    '43': 45,
    '44': 40,
    '52': 25,
    '53': 10,
    '54': 50,
    '83': 80,
}

BACKGROUND_VARS = ["mother_highest_grade", "father_highest_grade", "school_type", "regular_classroom"]

STD_VARS = ['total_math_score', 'total_verbal_score', 'total_memory_score']

STEM_VARS = [
    'lisas',
    'AWMA-S_VisuoSpatialSTM_StS',
    'AWMA-S_VisuoSpatialWM_StS',
    'CMAT_BasicCalc_Comp_Quotient',
    'KeyMath_Numeration_ScS',
    'KeyMath_Measurement_ScS',
    'KeyMath_ProblemSolving_ScS',
    'WASI_PIQ',
]

VERBAL_VARS = [
    'AWMA-S_VerbalSTM_StS',
    'AWMA-S_VerbalWM_StS',
    'CTOPP_PhonAwareness_Comp',
    'CTOPP_RapidNaming_Comp',
    'TOWRE_Total_StS',
    'WASI_VIQ',
]
