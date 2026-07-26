#JHAT params
strict_gaia_params = { 'telescope' : 'jwst',
                        'overwrite' : True,
                        'd2d_max' : 0.5,
                        'showplots' : 0,
                        'find_stars_threshold' : 5,
                        'iterate_with_xyshifts' : True,
                        'histocut_order' : 'dxdy',
                        'sharpness_lim' : (0.3,0.95),
                        'roundness1_lim' : (-0.7, 0.7),
                        'SNR_min' : 5,
                        'dmag_max' : 0.1,
                        'objmag_lim' : (15,25),
                        'slope_min' : -20/2048,
                        'binsize_px' : 1.0,
                        'savephottable' : 0 }

relaxed_gaia_params = { 'telescope' : 'jwst',
                        'overwrite' : True,
                        'd2d_max' : 2.0,
                        'showplots' : 0,
                        'find_stars_threshold' : 3,
                        'iterate_with_xyshifts' : False,
                        'histocut_order' : 'dxdy',
                        'sharpness_lim' : (0.3,0.95),
                        'roundness1_lim' : (-0.7, 0.7),
                        'SNR_min' : 3,
                        'dmag_max' : 0.1,
                        'slope_min' : -20/2048,
                        'binsize_px' : 1.0,
                        'savephottable' : 0 }

strict_jwst_params = { 'telescope' : 'jwst',
                        'refcat_racol' : 'ra',
                        'refcat_deccol' : 'dec',
                        'refcat_magcol' : 'mag',
                        'refcat_magerrcol' : 'dmag',
                        'overwrite' : True,
                        'd2d_max' : 0.5,
                        'showplots' : 0,
                        'find_stars_threshold' : 5,
                        'iterate_with_xyshifts' : True,
                        'histocut_order' : 'dxdy',
                        'sharpness_lim' : (0.3,0.95),
                        'roundness1_lim' : (-0.7, 0.7),
                        'SNR_min' : 5,
                        'dmag_max' : 0.1,
                        'objmag_lim' : (15,25),
                        'slope_min' : -20/2048,
                        'binsize_px' : 1.0,
                        'savephottable' : 0 }

relaxed_jwst_params = { 'telescope' : 'jwst',
                        'refcat_racol' : 'ra',
                        'refcat_deccol' : 'dec',
                        'refcat_magcol' : 'mag',
                        'refcat_magerrcol' : 'dmag',
                        'overwrite' : True,
                        'd2d_max' : 2.0,
                        'showplots' : 0,
                        'find_stars_threshold' : 3,
                        'iterate_with_xyshifts' : False,
                        'histocut_order' : 'dxdy',
                        'sharpness_lim' : (0.3,0.95),
                        'roundness1_lim' : (-0.7, 0.7),
                        'SNR_min' : 3,
                        'dmag_max' : 0.1,
                        'slope_min' : -20/2048,
                        'binsize_px' : 1.0,
                        'savephottable' : 0 }

# DOLPHOT params (NIRCam defaults used by dolphot_prep.setup_paramfile)
base_params = {'FitSky' : '2',
                'SigPSF' : '5.0',
                'FlagMask' : '4',
                'SecondPass' : '5',
                'PSFPhotIt' : '2',
                'ApCor' : '1',
                'FSat' : '0.999',
                'NoiseMult' : '0.1',
                'RCombine' : '1.5',
                'CombineChi' : '0',
                'MaxIT' : '25',
                'InterpPSFlib' : '1',
                'SigFindMult' : '0.85',
                'PSFPhot' : '1',
                'Force1' : '0',
                'SkySig' : '2.25',
                'SkipSky' : '1',
                'UseWCS' : '2',
                'PSFres' : '1',
                'PosStep' : '0.25',
                'NIRCAMvega' : '0',
                'Align' : '4',
                'aligntol' : '0',
                'Rotate' : '1'}

short_params = {'shift' : '0 0',
                'xform' :'1 0 0',
                'raper' : '2',
                'rchi' : '1.5',
                'rsky0' : '15',
                'rsky1' : '35',
                'rsky2' : '3 10',
                'rpsf' : '15',
                'apsky' : '20 35'}

long_params = {'shift' : '0 0',
                'xform' :'1 0 0',
                'raper' : '3',
                'rchi' : '2.0',
                'rsky0' : '15',
                'rsky1' : '35',
                'rsky2' : '4 10',
                'rpsf' : '15',
                'apsky' : '20 35'}

# MIRI per-image params for FitSky=2 (dolphotMIRI.pdf §4.1).
# img RAper=3, img RChi=2.0, img RSky=15 35, img RSky2=4 10, img RPSF=15,
# img apsky=20 35. RAper/RPSF cannot exceed 24 for MIRI.
miri_params = {
    'shift': '0 0',
    'xform': '1 0 0',
    'raper': '3',
    'rchi': '2.0',
    'rsky0': '15',
    'rsky1': '35',
    'rsky2': '4 10',
    'rpsf': '15',
    'apsky': '20 35',
}

# Global params when MIRI frames are present (UseWCS=2 required).
# MIRIvega=0 matches NIRCAMvega=0 (AB mag / Jy) used in NIRCam runs.
miri_base_params = {
    **base_params,
    'MIRIvega': '0',
    'RCentroid': '1',
}

# calcsky: NIRCam (existing mosaic defaults) vs MIRI (dolphotMIRI.pdf §3.4).
nircam_calcsky_params = {
    'rin': 15,
    'rout': 25,
    'step': -64,
    'sigma_low': 2.25,
    'sigma_high': 2.00,
}

miri_calcsky_params = {
    'rin': 10,
    'rout': 25,
    'step': -64,  # quick sky; sufficient with FitSky != 0
    'sigma_low': 2.25,
    'sigma_high': 2.00,
}

# Default DOLPHOT binary directory for this installation.
DEFAULT_DOLPHOT_BIN = '/data/software/dolphot/bin'