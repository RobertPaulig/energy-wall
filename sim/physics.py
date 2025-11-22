PHYSICS_CONFIG = {
    'RRAM': {
        'name': 'Standard RRAM (Filamentary HfO2)',
        'e_fJ_mxv': 150.0,
        'e_fJ_wire': 5.0,
        'e_fJ_adc': 50.0,
        'e_fJ_write': 500.0,
        'read_noise_std': 0.08,
        'rtn_fraction': 0.05,
        'rtn_amp': 0.15,
        'rtn_tau_on': 8,
        'rtn_tau_off': 8,
        'color': 'red'
    },
    'CDW_COHERENT': {
        'name': 'Coherent Phase (TaS2)',
        'e_fJ_mxv': 1.5,
        'e_fJ_wire': 1.0,
        'e_fJ_adc': 5.0,
        'e_fJ_write': 20.0,
        'read_noise_std': 0.005,
        'rtn_fraction': 0.0001,
        'rtn_amp': 0.01,
        'rtn_tau_on': 1000,
        'rtn_tau_off': 1000,
        'color': 'green'
    }
}