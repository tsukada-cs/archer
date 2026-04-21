import numpy as np


def confidence_to_alpha(confidence_score, archer_channel_type, fxHr, vmax):
    """
    Convert confidence score to alpha value.

    Parameters
    ----------
    confidence_score : float
        Confidence score between 0 and 100.
    archer_channel_type : str
        Type of sensor data.
    fxHr : float
        Forecast hour.
    vmax : float
        Maximum wind speed in knots.
    
    Returns
    -------
    alpha : float
        Alpha value.
    """
    alpha_floor = 0.5

    if archer_channel_type.lower() == 'rscat':
        m_fit = 1.8
        b_fit = -19.40
    elif archer_channel_type.lower() == 'ascat':
        m_fit = 2.67
        b_fit = -19.40
    elif archer_channel_type.lower() in ('89ghz', '37ghz', '183ghz'):
        m_fit_lo = 3.28
        b_fit_lo = 2.63
        m_fit_hi = 1.61
        b_fit_hi = 9.58
    elif archer_channel_type.lower() in ('ir', 'swir'):
        m_fit_lo = 8.68
        b_fit_lo = -0.37
        m_fit_hi = 14.2
        b_fit_hi = -0.24
    elif archer_channel_type.lower() in ('vis', 'dnb'):
        m_fit_lo = 14.44
        b_fit_lo = -0.83
        m_fit_hi = 14.64
        b_fit_hi = 3.45

    # Compute alpha_0 (alpha at dt=0) from the above parameters
    if archer_channel_type.lower() in ('rscat', 'ascat'):
        alpha_0 = np.max(m_fit * confidence_score + b_fit, alpha_floor)
    elif archer_channel_type.lower() in ('89ghz', '37ghz', '183ghz', 'ir', 'swir', 'vis', 'dnb'):
        alpha_lo0 = np.max([m_fit_lo * confidence_score + b_fit_lo, alpha_floor])
        alpha_hi0 = np.max([m_fit_hi * confidence_score + b_fit_hi, alpha_floor])
        alpha_0 = np.interp(vmax, [0, 60, 85, 300], [alpha_lo0, alpha_lo0, alpha_hi0, alpha_hi0])
    elif archer_channel_type.lower() == 'nhcanfx':
        alpha_at_48_kt = 9.5
        alpha_0 = alpha_at_48_kt + 0.05 * (vmax - 48)

    # Compute alpha at dt = fxHr for either analysis/forecast or imagery
    if archer_channel_type.lower() == 'nhcanfx':
        c1 = 3.00 / 13.4 / (15**1.5)
    else:
        c1 = 4.00 / 13.4 / (15**1.5)

    alpha = alpha_0 / (1 + c1 * alpha_0 * (fxHr**1.5))
    return alpha