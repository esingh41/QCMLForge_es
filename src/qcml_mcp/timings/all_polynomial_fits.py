dft_methods = [
    "B3LYP",
    "B3LYP-D3",
    "B2PLYP",
    "B2PLYP-D3",
    "B97",
    "wB97X-D",
    "M05-2X",
    "PBE",
    "PBE-D3",
    "B97-D3",
]
wfn_methods = [
    "MP2",
    "HF",
    "FNO-CCSD",
    "FNO-CCSD(T)",
    # "SAPT0",
    # "SAPT2",
]

fit_data = {
    "metadata": {"created": "polynomial_fitting.py"},
    "methods": {
        "B2PLYP": {
            "coefficients": [
                [-3.1440211363414563, 0.006865070648332795, -5.040959053078571e-06],
                [-2.8087118976138687, -0.001084371421939387, 3.257997025858611e-07],
                [0.12222307412309985, 4.7107857558374e-08],
                [0.6837797318001781, 0.00033598551003668673],
            ],
            "comment": "Coefficients correspond to features in the "
            "order: a00, a01*nocc^1, a02*nocc^2",
            "degrees": [2, 2, 1, 1],
            "feature_names": [
                ["a00", "a01*nocc^1", "a02*nocc^2"],
                ["a10", "a11*nvirt^1", "a12*nvirt^2"],
                ["a20", "a21*np_total^1"],
                ["a30", "a31*nbf_aux^1"],
            ],
            "latex_equation": "\\log[(-3.14 + 6.87 \\times 10^{-3} "
            "\\cdot N_{\\rm{occ}} - 5.04 \\times "
            "10^{-6} \\cdot N_{\\rm{occ}}^{2}) "
            "\\times (-2.81 - 1.08 \\times "
            "10^{-3} \\cdot N_{\\rm{virt}} + "
            "3.26 \\times 10^{-7} \\cdot "
            "N_{\\rm{virt}}^{2}) \\times (1.22 "
            "\\times 10^{-1} + 4.71 \\times "
            "10^{-8} \\cdot N_{\\rm{grid}}) "
            "\\times (6.84 \\times 10^{-1} + "
            "3.36 \\times 10^{-4} \\cdot "
            "N_{\\rm{aux}})]",
            "method": "B2PLYP",
            "method_name": "B2PLYP",
            "n_test_samples": 27,
            "n_train_samples": 243,
            "operators": ["*", "*", "*", "+"],
            "plot_output": "./plots/polyfit_B2PLYP.png",
            "test_error": {
                "mae": 0.08974427017458256,
                "mse": 0.12133950186295289,
                "r2": 0.9843349111949355,
            },
            "train_error": {
                "mae": 0.08925329632866794,
                "mse": 0.11421528345972883,
                "r2": 0.9850864283644825,
            },
            "variables": ["nocc", "nvirt", "np_total", "nbf_aux"],
            "y_variable": "$\\log_{10}[\\mathrm{time(s)}]$",
        },
        "B2PLYP-D3": {
            "coefficients": [
                [-3.1440211363414563, 0.006865070648332795, -5.040959053078571e-06],
                [-2.8087118976138687, -0.001084371421939387, 3.257997025858611e-07],
                [0.12222307412309985, 4.7107857558374e-08],
                [0.6837797318001781, 0.00033598551003668673],
            ],
            "comment": "Coefficients correspond to features in "
            "the order: a00, a01*nocc^1, a02*nocc^2",
            "degrees": [2, 2, 1, 1],
            "feature_names": [
                ["a00", "a01*nocc^1", "a02*nocc^2"],
                ["a10", "a11*nvirt^1", "a12*nvirt^2"],
                ["a20", "a21*np_total^1"],
                ["a30", "a31*nbf_aux^1"],
            ],
            "latex_equation": "\\log[(-3.14 + 6.87 \\times "
            "10^{-3} \\cdot N_{\\rm{occ}} - "
            "5.04 \\times 10^{-6} \\cdot "
            "N_{\\rm{occ}}^{2}) \\times "
            "(-2.81 - 1.08 \\times 10^{-3} "
            "\\cdot N_{\\rm{virt}} + 3.26 "
            "\\times 10^{-7} \\cdot "
            "N_{\\rm{virt}}^{2}) \\times "
            "(1.22 \\times 10^{-1} + 4.71 "
            "\\times 10^{-8} \\cdot "
            "N_{\\rm{grid}}) \\times (6.84 "
            "\\times 10^{-1} + 3.36 \\times "
            "10^{-4} \\cdot N_{\\rm{aux}})]",
            "method": "B2PLYP-D3",
            "method_name": "B2PLYP-D3",
            "n_test_samples": 27,
            "n_train_samples": 243,
            "operators": ["*", "*", "*", "+"],
            "plot_output": "./plots/polyfit_B2PLYP-D3.png",
            "test_error": {
                "mae": 0.08974427017458256,
                "mse": 0.12133950186295289,
                "r2": 0.9843349111949355,
            },
            "train_error": {
                "mae": 0.08925329632866794,
                "mse": 0.11421528345972883,
                "r2": 0.9850864283644825,
            },
            "variables": ["nocc", "nvirt", "np_total", "nbf_aux"],
            "y_variable": "$\\log_{10}[\\mathrm{time(s)}]$",
        },
        "B3LYP": {
            "coefficients": [
                [-2.394027814520387, 0.007307583672168582, -7.983251989976164e-06],
                [-1.907978033184169, -0.000574512365907133, 1.6052357228367702e-07],
                [0.3683270074465912, 1.9803039892041405e-07],
                [0.3973089773146867, 0.00023721612483082021],
            ],
            "comment": "Coefficients correspond to features in the "
            "order: a00, a01*nocc^1, a02*nocc^2",
            "degrees": [2, 2, 1, 1],
            "feature_names": [
                ["a00", "a01*nocc^1", "a02*nocc^2"],
                ["a10", "a11*nvirt^1", "a12*nvirt^2"],
                ["a20", "a21*np_total^1"],
                ["a30", "a31*nbf_aux^1"],
            ],
            "latex_equation": "\\log[(-2.39 + 7.31 \\times 10^{-3} "
            "\\cdot N_{\\rm{occ}} - 7.98 \\times "
            "10^{-6} \\cdot N_{\\rm{occ}}^{2}) "
            "\\times (-1.91 - 5.75 \\times "
            "10^{-4} \\cdot N_{\\rm{virt}} + 1.61 "
            "\\times 10^{-7} \\cdot "
            "N_{\\rm{virt}}^{2}) \\times (3.68 "
            "\\times 10^{-1} + 1.98 \\times "
            "10^{-7} \\cdot N_{\\rm{grid}}) "
            "\\times (3.97 \\times 10^{-1} + 2.37 "
            "\\times 10^{-4} \\cdot "
            "N_{\\rm{aux}})]",
            "method": "B3LYP",
            "method_name": "B3LYP",
            "n_test_samples": 23,
            "n_train_samples": 201,
            "operators": ["*", "*", "*", "+"],
            "plot_output": "./plots/polyfit_B3LYP.png",
            "test_error": {
                "mae": 0.12311187477911614,
                "mse": 0.15799279564060373,
                "r2": 0.9724100356212982,
            },
            "train_error": {
                "mae": 0.09313374069013099,
                "mse": 0.12974402880638122,
                "r2": 0.9800959172530193,
            },
            "variables": ["nocc", "nvirt", "np_total", "nbf_aux"],
            "y_variable": "$\\log_{10}[\\mathrm{time(s)}]$",
        },
        "B3LYP-D3": {
            "coefficients": [
                [-2.394027814520387, 0.007307583672168582, -7.983251989976164e-06],
                [-1.907978033184169, -0.000574512365907133, 1.6052357228367702e-07],
                [0.3683270074465912, 1.9803039892041405e-07],
                [0.3973089773146867, 0.00023721612483082021],
            ],
            "comment": "Coefficients correspond to features in "
            "the order: a00, a01*nocc^1, a02*nocc^2",
            "degrees": [2, 2, 1, 1],
            "feature_names": [
                ["a00", "a01*nocc^1", "a02*nocc^2"],
                ["a10", "a11*nvirt^1", "a12*nvirt^2"],
                ["a20", "a21*np_total^1"],
                ["a30", "a31*nbf_aux^1"],
            ],
            "latex_equation": "\\log[(-2.39 + 7.31 \\times "
            "10^{-3} \\cdot N_{\\rm{occ}} - "
            "7.98 \\times 10^{-6} \\cdot "
            "N_{\\rm{occ}}^{2}) \\times (-1.91 "
            "- 5.75 \\times 10^{-4} \\cdot "
            "N_{\\rm{virt}} + 1.61 \\times "
            "10^{-7} \\cdot "
            "N_{\\rm{virt}}^{2}) \\times (3.68 "
            "\\times 10^{-1} + 1.98 \\times "
            "10^{-7} \\cdot N_{\\rm{grid}}) "
            "\\times (3.97 \\times 10^{-1} + "
            "2.37 \\times 10^{-4} \\cdot "
            "N_{\\rm{aux}})]",
            "method": "B3LYP-D3",
            "method_name": "B3LYP-D3",
            "n_test_samples": 23,
            "n_train_samples": 201,
            "operators": ["*", "*", "*", "+"],
            "plot_output": "./plots/polyfit_B3LYP-D3.png",
            "test_error": {
                "mae": 0.12311187477911614,
                "mse": 0.15799279564060373,
                "r2": 0.9724100356212982,
            },
            "train_error": {
                "mae": 0.09313374069013099,
                "mse": 0.12974402880638122,
                "r2": 0.9800959172530193,
            },
            "variables": ["nocc", "nvirt", "np_total", "nbf_aux"],
            "y_variable": "$\\log_{10}[\\mathrm{time(s)}]$",
        },
        "FNO-CCSD": {
            "coefficients": [
                [
                    1.245754752909329,
                    0.18532340162224037,
                    -0.005750544631096848,
                    9.635704127343583e-05,
                    -5.550011795668423e-07,
                ],
                [-0.8823692537289531, 0.11369228067555259, -0.00011777228352299516],
                [0.06088855966739218, -2.63997556710382e-05],
            ],
            "comment": "Coefficients correspond to features in "
            "the order: a00, a01*nocc^1, a02*nocc^2, "
            "a03*nocc^3, a04*nocc^4",
            "degrees": [4, 2, 1],
            "feature_names": [
                ["a00", "a01*nocc^1", "a02*nocc^2", "a03*nocc^3", "a04*nocc^4"],
                ["a10", "a11*nvirt^1", "a12*nvirt^2"],
                ["a20", "a21*nbf_aux^1"],
            ],
            "latex_equation": "\\log[(1.25 + 1.85 \\times "
            "10^{-1} \\cdot N_{\\rm{occ}} - "
            "5.75 \\times 10^{-3} \\cdot "
            "N_{\\rm{occ}}^{2} + 9.64 \\times "
            "10^{-5} \\cdot N_{\\rm{occ}}^{3} "
            "- 5.55 \\times 10^{-7} \\cdot "
            "N_{\\rm{occ}}^{4}) \\times (-8.82 "
            "\\times 10^{-1} + 1.14 \\times "
            "10^{-1} \\cdot N_{\\rm{virt}} - "
            "1.18 \\times 10^{-4} \\cdot "
            "N_{\\rm{virt}}^{2}) \\times (6.09 "
            "\\times 10^{-2} - 2.64 \\times "
            "10^{-5} \\cdot N_{\\rm{aux}})]",
            "method": "FNO-CCSD",
            "method_name": "FNO-CCSD",
            "n_test_samples": 4,
            "n_train_samples": 32,
            "operators": ["*", "*", "*"],
            "plot_output": "./plots/polyfit_FNO-CCSD.png",
            "test_error": {
                "mae": 0.21318003281134418,
                "mse": 0.30010842605870947,
                "r2": 0.8118001848521468,
            },
            "train_error": {
                "mae": 0.06472725253538725,
                "mse": 0.07564253536014807,
                "r2": 0.9925860873059035,
            },
            "variables": ["nocc", "nvirt", "nbf_aux"],
            "y_variable": "$\\log_{10}[\\mathrm{time(s)}]$",
        },
        "FNO-CCSD(T)": {
            "coefficients": [
                [
                    -5.791845519136249,
                    0.10169366224430088,
                    -0.0009058644615177612,
                    -5.790178730437877e-06,
                    8.617165703405521e-08,
                ],
                [
                    -3.9685814775903427,
                    -0.0163719831817734,
                    6.748478266292185e-05,
                    -7.422645415650154e-08,
                ],
                [-0.019372932007445873, 0.00025979435400215586],
            ],
            "comment": "Coefficients correspond to features "
            "in the order: a00, a01*nocc^1, "
            "a02*nocc^2, a03*nocc^3, a04*nocc^4",
            "degrees": [4, 3, 1],
            "feature_names": [
                ["a00", "a01*nocc^1", "a02*nocc^2", "a03*nocc^3", "a04*nocc^4"],
                ["a10", "a11*nvirt^1", "a12*nvirt^2", "a13*nvirt^3"],
                ["a20", "a21*nbf_aux^1"],
            ],
            "latex_equation": "\\log[(-5.79 + 1.02 \\times "
            "10^{-1} \\cdot N_{\\rm{occ}} - "
            "9.06 \\times 10^{-4} \\cdot "
            "N_{\\rm{occ}}^{2} - 5.79 "
            "\\times 10^{-6} \\cdot "
            "N_{\\rm{occ}}^{3} + 8.62 "
            "\\times 10^{-8} \\cdot "
            "N_{\\rm{occ}}^{4}) \\times "
            "(-3.97 - 1.64 \\times 10^{-2} "
            "\\cdot N_{\\rm{virt}} + 6.75 "
            "\\times 10^{-5} \\cdot "
            "N_{\\rm{virt}}^{2} - 7.42 "
            "\\times 10^{-8} \\cdot "
            "N_{\\rm{virt}}^{3}) \\times "
            "(-1.94 \\times 10^{-2} + 2.60 "
            "\\times 10^{-4} \\cdot "
            "N_{\\rm{aux}})]",
            "method": "FNO-CCSD(T)",
            "method_name": "FNO-CCSD(T)",
            "n_test_samples": 4,
            "n_train_samples": 31,
            "operators": ["*", "*", "*"],
            "plot_output": "./plots/polyfit_FNO-CCSD(T).png",
            "test_error": {
                "mae": 0.10966148515110341,
                "mse": 0.12187988016332343,
                "r2": 0.9347584060664296,
            },
            "train_error": {
                "mae": 0.06951337024343368,
                "mse": 0.08078273260853859,
                "r2": 0.9943904832083275,
            },
            "variables": ["nocc", "nvirt", "nbf_aux"],
            "y_variable": "$\\log_{10}[\\mathrm{time(s)}]$",
        },
        "HF": {
            "coefficients": [
                [-4.087184018422912, 0.003572877739081107, -1.2607644427263935e-06],
                [-3.8285623877134873, 0.0005409974625158854, -9.078005060787753e-08],
                [-0.0023771122598862217, 5.524109894744721e-05],
            ],
            "comment": "Coefficients correspond to features in the "
            "order: a00, a01*nocc^1, a02*nocc^2",
            "degrees": [2, 2, 1],
            "feature_names": [
                ["a00", "a01*nocc^1", "a02*nocc^2"],
                ["a10", "a11*nvirt^1", "a12*nvirt^2"],
                ["a20", "a21*nbf_aux^1"],
            ],
            "latex_equation": "\\log[(-4.09 + 3.57 \\times 10^{-3} "
            "\\cdot N_{\\rm{occ}} - 1.26 \\times "
            "10^{-6} \\cdot N_{\\rm{occ}}^{2}) "
            "\\times (-3.83 + 5.41 \\times 10^{-4} "
            "\\cdot N_{\\rm{virt}} - 9.08 \\times "
            "10^{-8} \\cdot N_{\\rm{virt}}^{2}) "
            "\\times (-2.38 \\times 10^{-3} + 5.52 "
            "\\times 10^{-5} \\cdot N_{\\rm{aux}})]",
            "method": "HF",
            "method_name": "HF",
            "n_test_samples": 14,
            "n_train_samples": 126,
            "operators": ["*", "*", "*"],
            "plot_output": "./plots/polyfit_HF.png",
            "test_error": {
                "mae": 0.0774746919207481,
                "mse": 0.10045076865792293,
                "r2": 0.9819196718341884,
            },
            "train_error": {
                "mae": 0.09180469330331066,
                "mse": 0.11735959358333713,
                "r2": 0.9867114399372467,
            },
            "variables": ["nocc", "nvirt", "nbf_aux"],
            "y_variable": "$\\log_{10}[\\mathrm{time(s)}]$",
        },
        "M05-2X": {
            "coefficients": [
                [
                    -5.630373415388223,
                    -8.376114197433675e-05,
                    -9.34974505779708e-07,
                    5.000204395774288e-10,
                    -7.36962574299227e-14,
                ],
                [-4.20405884143134, 3.953153656582772e-07],
                [0.03513255624065878, 2.1966497018143082e-05],
            ],
            "comment": "Coefficients correspond to features in the "
            "order: a00, a01*nbf^1, a02*nbf^2, "
            "a03*nbf^3, a04*nbf^4",
            "degrees": [4, 1, 1],
            "feature_names": [
                ["a00", "a01*nbf^1", "a02*nbf^2", "a03*nbf^3", "a04*nbf^4"],
                ["a10", "a11*np_total^1"],
                ["a20", "a21*nbf_aux^1"],
            ],
            "latex_equation": "\\log[(-5.63 - 8.38 \\times 10^{-5} "
            "\\cdot N_{\\rm{bf}} - 9.35 \\times "
            "10^{-7} \\cdot N_{\\rm{bf}}^{2} + "
            "5.00 \\times 10^{-10} \\cdot "
            "N_{\\rm{bf}}^{3} - 7.37 \\times "
            "10^{-14} \\cdot N_{\\rm{bf}}^{4}) "
            "\\times (-4.20 + 3.95 \\times "
            "10^{-7} \\cdot N_{\\rm{grid}}) "
            "\\times (3.51 \\times 10^{-2} + "
            "2.20 \\times 10^{-5} \\cdot "
            "N_{\\rm{aux}})]",
            "method": "M05-2X",
            "method_name": "M05-2X",
            "n_test_samples": 22,
            "n_train_samples": 195,
            "operators": ["*", "*", "+"],
            "plot_output": "./plots/polyfit_M05-2X.png",
            "test_error": {
                "mae": 0.15255364665190208,
                "mse": 0.18070211520412696,
                "r2": 0.9664117423493715,
            },
            "train_error": {
                "mae": 0.14489672152256255,
                "mse": 0.18072508377554197,
                "r2": 0.9589491723451935,
            },
            "variables": ["nbf", "np_total", "nbf_aux"],
            "y_variable": "$\\log_{10}[\\mathrm{time(s)}]$",
        },
        "MP2": {
            "coefficients": [
                [0.2759308904810978, 0.0018543185069818838, 2.8122931511922745e-07],
                [1.2308226500020585, 0.0009379926543264187, 1.8713565944737797e-07],
                [2.4440806047657877, -0.00017926558019083785],
            ],
            "comment": "Coefficients correspond to features in the "
            "order: a00, a01*nocc^1, a02*nocc^2",
            "degrees": [2, 2, 1],
            "feature_names": [
                ["a00", "a01*nocc^1", "a02*nocc^2"],
                ["a10", "a11*nvirt^1", "a12*nvirt^2"],
                ["a20", "a21*nbf_aux^1"],
            ],
            "latex_equation": "\\log[(2.76 \\times 10^{-1} + 1.85 "
            "\\times 10^{-3} \\cdot N_{\\rm{occ}} + "
            "2.81 \\times 10^{-7} \\cdot "
            "N_{\\rm{occ}}^{2}) \\times (1.23 + "
            "9.38 \\times 10^{-4} \\cdot "
            "N_{\\rm{virt}} + 1.87 \\times 10^{-7} "
            "\\cdot N_{\\rm{virt}}^{2}) \\times "
            "(2.44 - 1.79 \\times 10^{-4} \\cdot "
            "N_{\\rm{aux}})]",
            "method": "MP2",
            "method_name": "MP2",
            "n_test_samples": 24,
            "n_train_samples": 210,
            "operators": ["*", "*", "*", "*"],
            "plot_output": "./plots/polyfit_MP2.png",
            "test_error": {
                "mae": 0.15753838753912816,
                "mse": 0.1855295841452476,
                "r2": 0.968545000105604,
            },
            "train_error": {
                "mae": 0.12028962249615195,
                "mse": 0.15417912935505582,
                "r2": 0.9682305208037436,
            },
            "variables": ["nocc", "nvirt", "nbf_aux"],
            "y_variable": "$\\log_{10}[\\mathrm{time(s)}]$",
        },
        "PBE": {
            "coefficients": [
                [
                    -4.877660821671894,
                    0.010177013009294916,
                    -1.662263221404033e-05,
                    1.360448697812041e-08,
                ],
                [-1.8969867273052874, -5.5083153491836355e-05],
                [0.7739749266582981, 2.532482531436583e-08],
                [0.08868378791981453, 9.470135563727018e-05],
            ],
            "comment": "Coefficients correspond to features in the "
            "order: a00, a01*nocc^1, a02*nocc^2, "
            "a03*nocc^3",
            "degrees": [3, 1, 1, 1],
            "feature_names": [
                ["a00", "a01*nocc^1", "a02*nocc^2", "a03*nocc^3"],
                ["a10", "a11*nvirt^1"],
                ["a20", "a21*np_total^1"],
                ["a30", "a31*nbf_aux^1"],
            ],
            "latex_equation": "\\log[(-4.88 + 1.02 \\times 10^{-2} "
            "\\cdot N_{\\rm{occ}} - 1.66 \\times "
            "10^{-5} \\cdot N_{\\rm{occ}}^{2} + "
            "1.36 \\times 10^{-8} \\cdot "
            "N_{\\rm{occ}}^{3}) \\times (-1.90 - "
            "5.51 \\times 10^{-5} \\cdot "
            "N_{\\rm{virt}}) \\times (7.74 \\times "
            "10^{-1} + 2.53 \\times 10^{-8} \\cdot "
            "N_{\\rm{grid}}) \\times (8.87 \\times "
            "10^{-2} + 9.47 \\times 10^{-5} \\cdot "
            "N_{\\rm{aux}})]",
            "method": "PBE",
            "method_name": "PBE",
            "n_test_samples": 46,
            "n_train_samples": 408,
            "operators": ["*", "*", "*", "+"],
            "plot_output": "./plots/polyfit_PBE.png",
            "test_error": {
                "mae": 0.12397249707124955,
                "mse": 0.15543472544079817,
                "r2": 0.967747438704534,
            },
            "train_error": {
                "mae": 0.13423098251970209,
                "mse": 0.16381703445210333,
                "r2": 0.9630442868975335,
            },
            "variables": ["nocc", "nvirt", "np_total", "nbf_aux"],
            "y_variable": "$\\log_{10}[\\mathrm{time(s)}]$",
        },
        "PBE-D3": {
            "coefficients": [
                [
                    -4.877660821671894,
                    0.010177013009294916,
                    -1.662263221404033e-05,
                    1.360448697812041e-08,
                ],
                [-1.8969867273052874, -5.5083153491836355e-05],
                [0.7739749266582981, 2.532482531436583e-08],
                [0.08868378791981453, 9.470135563727018e-05],
            ],
            "comment": "Coefficients correspond to features in the "
            "order: a00, a01*nocc^1, a02*nocc^2, "
            "a03*nocc^3",
            "degrees": [3, 1, 1, 1],
            "feature_names": [
                ["a00", "a01*nocc^1", "a02*nocc^2", "a03*nocc^3"],
                ["a10", "a11*nvirt^1"],
                ["a20", "a21*np_total^1"],
                ["a30", "a31*nbf_aux^1"],
            ],
            "latex_equation": "\\log[(-4.88 + 1.02 \\times 10^{-2} "
            "\\cdot N_{\\rm{occ}} - 1.66 \\times "
            "10^{-5} \\cdot N_{\\rm{occ}}^{2} + "
            "1.36 \\times 10^{-8} \\cdot "
            "N_{\\rm{occ}}^{3}) \\times (-1.90 - "
            "5.51 \\times 10^{-5} \\cdot "
            "N_{\\rm{virt}}) \\times (7.74 "
            "\\times 10^{-1} + 2.53 \\times "
            "10^{-8} \\cdot N_{\\rm{grid}}) "
            "\\times (8.87 \\times 10^{-2} + "
            "9.47 \\times 10^{-5} \\cdot "
            "N_{\\rm{aux}})]",
            "method": "PBE-D3",
            "method_name": "PBE-D3",
            "n_test_samples": 46,
            "n_train_samples": 408,
            "operators": ["*", "*", "*", "+"],
            "plot_output": "./plots/polyfit_PBE-D3.png",
            "test_error": {
                "mae": 0.12397249707124955,
                "mse": 0.15543472544079817,
                "r2": 0.967747438704534,
            },
            "train_error": {
                "mae": 0.13423098251970209,
                "mse": 0.16381703445210333,
                "r2": 0.9630442868975335,
            },
            "variables": ["nocc", "nvirt", "np_total", "nbf_aux"],
            "y_variable": "$\\log_{10}[\\mathrm{time(s)}]$",
        },
        "wB97X-D": {
            "coefficients": [
                [
                    2.0241543257375114,
                    0.0045776361580094,
                    -5.020953785281088e-06,
                    2.2936782424722992e-09,
                    -3.6614157886138976e-13,
                ],
                [6.753190561963754, -7.395071174847531e-07],
                [0.033517707220830605, 3.241351630111042e-05],
            ],
            "comment": "Coefficients correspond to features in "
            "the order: a00, a01*nbf^1, a02*nbf^2, "
            "a03*nbf^3, a04*nbf^4",
            "degrees": [4, 1, 1],
            "feature_names": [
                ["a00", "a01*nbf^1", "a02*nbf^2", "a03*nbf^3", "a04*nbf^4"],
                ["a10", "a11*np_total^1"],
                ["a20", "a21*nbf_aux^1"],
            ],
            "latex_equation": "\\log[(2.02 + 4.58 \\times 10^{-3} "
            "\\cdot N_{\\rm{bf}} - 5.02 \\times "
            "10^{-6} \\cdot N_{\\rm{bf}}^{2} + "
            "2.29 \\times 10^{-9} \\cdot "
            "N_{\\rm{bf}}^{3} - 3.66 \\times "
            "10^{-13} \\cdot N_{\\rm{bf}}^{4}) "
            "\\times (6.75 - 7.40 \\times "
            "10^{-7} \\cdot N_{\\rm{grid}}) "
            "\\times (3.35 \\times 10^{-2} + "
            "3.24 \\times 10^{-5} \\cdot "
            "N_{\\rm{aux}})]",
            "method": "wB97X-D",
            "method_name": "wB97X-D",
            "n_test_samples": 21,
            "n_train_samples": 182,
            "operators": ["*", "*", "+"],
            "plot_output": "./plots/polyfit_wB97X-D.png",
            "test_error": {
                "mae": 0.15339598560078277,
                "mse": 0.17705005036281976,
                "r2": 0.9729660254143987,
            },
            "train_error": {
                "mae": 0.11510049750960494,
                "mse": 0.15126891493665187,
                "r2": 0.9742170496781816,
            },
            "variables": ["nbf", "np_total", "nbf_aux"],
            "y_variable": "$\\log_{10}[\\mathrm{time(s)}]$",
        },
        "wB97X-V": {
            "coefficients": [
                [2.4381186808329143, 0.04680345473545202, -6.178211275079922e-05],
                [0.4094406601572704, -3.9060543448318675e-07, 8.881698788635312e-09],
                [4.338426355597796, -6.154823107242462e-07],
                [0.27249160065543027, 1.1629577257059353e-05],
            ],
            "comment": "Coefficients correspond to features in "
            "the order: a00, a01*nocc^1, a02*nocc^2",
            "degrees": [2, 2, 1, 1],
            "feature_names": [
                ["a00", "a01*nocc^1", "a02*nocc^2"],
                ["a10", "a11*nvirt^1", "a12*nvirt^2"],
                ["a20", "a21*np_total^1"],
                ["a30", "a31*nbf_aux^1"],
            ],
            "latex_equation": "\\log[(2.44 + 4.68 \\times 10^{-2} "
            "\\cdot N_{\\rm{occ}} - 6.18 "
            "\\times 10^{-5} \\cdot "
            "N_{\\rm{occ}}^{2}) \\times (4.09 "
            "\\times 10^{-1} - 3.91 \\times "
            "10^{-7} \\cdot N_{\\rm{virt}} + "
            "8.88 \\times 10^{-9} \\cdot "
            "N_{\\rm{virt}}^{2}) \\times (4.34 "
            "- 6.15 \\times 10^{-7} \\cdot "
            "N_{\\rm{grid}}) \\times (2.72 "
            "\\times 10^{-1} + 1.16 \\times "
            "10^{-5} \\cdot N_{\\rm{aux}})]",
            "method": "wB97X-V",
            "method_name": "wB97X-V",
            "n_test_samples": 23,
            "n_train_samples": 199,
            "operators": ["*", "*", "*", "+"],
            "plot_output": "./plots/polyfit_wB97X-V.png",
            "test_error": {
                "mae": 0.07054578233548094,
                "mse": 0.08544554542301415,
                "r2": 0.9902371137613144,
            },
            "train_error": {
                "mae": 0.11751837322059758,
                "mse": 0.15675855268726915,
                "r2": 0.9703007330202956,
            },
            "variables": ["nocc", "nvirt", "np_total", "nbf_aux"],
            "y_variable": "$\\log_{10}[\\mathrm{time(s)}]$",
        },
    },
}
