import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from meteostat import daily
import requests

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Genesee River Ice Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CONSTANTS ---
GENESEE_USGS_STATION = "04231600"  # Genesee River at Ford St, Rochester
KROC_STATION_ID = "72529"  # Rochester International Airport

# --- 1. ROBUST DATA FETCHING ---
@st.cache_data
def get_rochester_data(start_year=2000):
    """Fetch weather data from Rochester Airport (KROC)."""
    start = datetime(start_year, 1, 1)
    end = datetime.now()

    # Fetch from KROC station
    try:
        ts = daily(KROC_STATION_ID, start, end)
        data = ts.fetch()
    except Exception:
        return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    data = data.ffill()

    # Temperature Standardization (C to F)
    if 'tmin' in data.columns:
        data['tmin_f'] = data['tmin'] * 9/5 + 32
    if 'tmax' in data.columns:
        data['tmax_f'] = data['tmax'] * 9/5 + 32

    # Robust Average Temp Calculation
    if 'tavg' in data.columns:
        data['tavg_f'] = data['tavg'] * 9/5 + 32
    elif 'temp' in data.columns:
        data['tavg_f'] = data['temp'] * 9/5 + 32
    elif 'tmin' in data.columns and 'tmax' in data.columns:
        data['tavg_f'] = ((data['tmin'] + data['tmax']) / 2) * 9/5 + 32
    else:
        return pd.DataFrame()

    # Wind speed (m/s to mph) - critical for ice formation
    if 'wspd' in data.columns:
        data['wspd_mph'] = data['wspd'] * 2.237
    else:
        data['wspd_mph'] = 5.0  # Default assumption

    # Precipitation (mm)
    if 'prcp' not in data.columns:
        data['prcp'] = 0.0

    # Snow depth (mm) if available
    if 'snow' not in data.columns:
        data['snow'] = 0.0

    # Solar proxy: day of year for declination angle estimation
    data['doy'] = data.index.dayofyear
    # Simplified solar factor: peaks at summer solstice (172), lowest at winter (355/Dec 21)
    # Range from ~0.3 (Dec) to ~1.0 (June)
    data['solar_factor'] = 0.3 + 0.7 * (1 + np.cos(2 * np.pi * (data['doy'] - 172) / 365)) / 2

    return data


@st.cache_data
def get_usgs_flow_data(start_year=2000):
    """
    Fetch river discharge data from USGS for the Genesee River.
    Higher flow = more turbulence = harder for ice to form.
    """
    start = f"{start_year}-01-01"
    end = datetime.now().strftime("%Y-%m-%d")

    url = (
        f"https://waterservices.usgs.gov/nwis/dv/"
        f"?format=json&sites={GENESEE_USGS_STATION}"
        f"&startDT={start}&endDT={end}"
        f"&parameterCd=00060"  # Discharge in cubic feet per second
        f"&siteStatus=all"
    )

    try:
        response = requests.get(url, timeout=30)
        data = response.json()

        values = data['value']['timeSeries'][0]['values'][0]['value']

        records = []
        for v in values:
            records.append({
                'date': pd.to_datetime(v['dateTime']).tz_localize(None),
                'discharge_cfs': float(v['value']) if v['value'] != '-999999' else np.nan
            })

        df = pd.DataFrame(records)
        df = df.set_index('date')
        df = df.ffill().bfill()

        # Normalize discharge to a factor (1.0 = median flow, higher = more turbulence)
        median_flow = df['discharge_cfs'].median()
        df['flow_factor'] = df['discharge_cfs'] / median_flow
        df['flow_factor'] = df['flow_factor'].clip(0.5, 3.0)  # Reasonable bounds

        return df
    except Exception as e:
        return pd.DataFrame()

# --- 2. LOGIC ENGINES ---

# Ice thickness threshold for closure (inches)
ICE_CLOSURE_THRESHOLD = 2.5  # Aluminum hull safety limit

# Water heat bank constants
# Represents ~3-5 days of sub-freezing temps needed to cool water mass before ice forms
DEFAULT_HEAT_BANK = 35  # Degree-days of thermal mass before freezing can begin (rivers cool faster than lakes)
MAX_HEAT_BANK = 75  # Maximum heat bank capacity (doesn't fully reset in short thaws)
HEAT_BANK_RECHARGE_RATE = 0.3  # How fast heat bank refills during thaw (slower = more realistic)


def calculate_linear_fdd_model(df, fdd_limit, melt_factor, base_temp, flow_df=None):
    """
    Original Linear Cumulative Freezing Degree Days (FDD) Model.
    Simple but less physically accurate.
    """
    df = df.copy()
    temps = df['tavg_f'].values

    ice_severity = np.zeros(len(df))
    is_closed = np.zeros(len(df))
    current_ice = 0

    for i in range(len(df)):
        avg_temp = temps[i]

        if avg_temp < base_temp:
            degree_days = base_temp - avg_temp
            current_ice += degree_days
        else:
            degree_days = avg_temp - base_temp
            current_ice -= (degree_days * melt_factor)

        if current_ice < 0:
            current_ice = 0
        ice_severity[i] = current_ice

        if current_ice >= fdd_limit:
            is_closed[i] = 1

    df['Closed'] = is_closed
    df['Metric'] = ice_severity
    df['IceThickness'] = ice_severity / fdd_limit * ICE_CLOSURE_THRESHOLD
    return df


def calculate_stefan_model(df, alpha=0.78, melt_rate=0.12, base_temp=32, flow_df=None,
                           initial_heat_bank=DEFAULT_HEAT_BANK):
    """
    Stefan Equation Model - Physically accurate ice growth with thermal lag.

    Ice thickness follows: h = α × √(FDD)

    Where:
    - α (alpha) = Stefan coefficient (~0.5-0.7 for rivers, 0.8-1.0 for calm lakes)
    - FDD = cumulative freezing degree days

    This model captures:
    - Diminishing returns: first 10 FDD grows more ice than second 10 FDD
    - Water heat bank: water must cool before ice can form (thermal lag)
    - Flow effects: turbulence reduces ice formation
    """
    df = df.copy()
    temps = df['tavg_f'].values

    # Get flow factors if available
    flow_factors = np.ones(len(df))
    if flow_df is not None and not flow_df.empty:
        for i, idx in enumerate(df.index):
            if idx in flow_df.index:
                flow_factors[i] = flow_df.loc[idx, 'flow_factor']

    ice_thickness = np.zeros(len(df))  # inches
    is_closed = np.zeros(len(df))
    water_status = []  # Track water thermal state
    cumulative_fdd = 0
    current_thickness = 0.0
    water_heat_bank = initial_heat_bank  # Thermal mass of water

    for i in range(len(df)):
        avg_temp = temps[i]
        flow = flow_factors[i]

        if avg_temp < base_temp:
            # Cold day - either drain heat bank or grow ice
            daily_fdd = base_temp - avg_temp
            daily_fdd = daily_fdd / (flow ** 0.5)  # Flow reduces cooling

            if water_heat_bank > 0:
                # Water still has thermal mass - drain it first
                water_heat_bank -= daily_fdd
                if water_heat_bank < 0:
                    # Overflow goes to ice formation
                    overflow = -water_heat_bank
                    water_heat_bank = 0
                    cumulative_fdd += overflow
                    current_thickness = alpha * np.sqrt(cumulative_fdd)
                water_status.append('Cooling')
            else:
                # Water is at freezing - grow ice
                cumulative_fdd += daily_fdd
                current_thickness = alpha * np.sqrt(cumulative_fdd)
                water_status.append('Freezing')
        else:
            # Warm day - melt ice or recharge heat bank
            melt_degrees = avg_temp - base_temp
            effective_melt = melt_degrees * melt_rate * (flow ** 0.3)

            if current_thickness > 0:
                # Melt existing ice first
                current_thickness -= effective_melt
                if current_thickness < 0:
                    current_thickness = 0
                    cumulative_fdd = 0
                else:
                    cumulative_fdd = (current_thickness / alpha) ** 2
                water_status.append('Thawing')
            else:
                # No ice - recharge heat bank
                recharge = melt_degrees * HEAT_BANK_RECHARGE_RATE
                water_heat_bank = min(water_heat_bank + recharge, MAX_HEAT_BANK)
                water_status.append('Warming')

        ice_thickness[i] = current_thickness

        if current_thickness >= ICE_CLOSURE_THRESHOLD:
            is_closed[i] = 1

    df['Closed'] = is_closed
    df['Metric'] = ice_thickness
    df['IceThickness'] = ice_thickness
    df['WaterStatus'] = water_status
    return df


def calculate_modified_degree_day(df, alpha=0.78, base_melt=0.12, base_temp=32,
                                   use_wind=True, use_solar=True, use_snow=True,
                                   flow_df=None, initial_heat_bank=DEFAULT_HEAT_BANK):
    """
    Modified Degree Day Model - Includes environmental corrections and thermal lag.

    Adjustments:
    - Water heat bank: Thermal lag before ice can form
    - Wind: High wind prevents sheet ice formation (waves break up ice)
    - Solar: February sun melts ice even at 30°F
    - Snow: Insulates ice from cold air, slowing growth
    - Flow: River turbulence from USGS discharge data
    - Precipitation: Rain accelerates melt
    """
    df = df.copy()
    temps = df['tavg_f'].values
    wind = df['wspd_mph'].values if use_wind else np.full(len(df), 5.0)
    solar = df['solar_factor'].values if use_solar else np.full(len(df), 0.5)
    precip = df['prcp'].values
    snow = df['snow'].values if 'snow' in df.columns else np.zeros(len(df))

    # Get flow factors if available
    flow_factors = np.ones(len(df))
    if flow_df is not None and not flow_df.empty:
        for i, idx in enumerate(df.index):
            if idx in flow_df.index:
                flow_factors[i] = flow_df.loc[idx, 'flow_factor']

    ice_thickness = np.zeros(len(df))
    is_closed = np.zeros(len(df))
    water_status = []
    cumulative_fdd = 0
    current_thickness = 0.0
    water_heat_bank = initial_heat_bank

    for i in range(len(df)):
        avg_temp = temps[i]
        w = wind[i]
        s = solar[i]
        p = precip[i]
        snow_depth = snow[i]  # mm
        flow = flow_factors[i]

        # Wind factor: calm (< 5 mph) = 1.0, mild reduction at higher winds
        # Capped at 30% reduction - wind also chills water, partially offsetting disruption
        wind_factor = max(0.7, 1.0 - (w - 5) * 0.02) if use_wind else 1.0

        # Snow insulation factor: snow on ice slows further growth
        # snow_depth is in mm, convert to inches for threshold
        snow_inches = snow_depth / 25.4
        if use_snow and snow_inches > 1.0 and current_thickness > 0:
            # Snow insulates - reduce freezing rate exponentially with depth
            # 1" snow = 50% reduction, 3" = 75%, 6" = 87%
            snow_factor = 1.0 / (1.0 + snow_inches * 0.5)
        else:
            snow_factor = 1.0

        if avg_temp < base_temp:
            daily_fdd = base_temp - avg_temp

            # Apply corrections to freezing rate
            daily_fdd *= wind_factor  # Wind disrupts ice formation
            daily_fdd *= snow_factor  # Snow insulates
            daily_fdd /= (flow ** 0.5)  # Turbulence from flow

            if water_heat_bank > 0:
                # Drain heat bank first
                water_heat_bank -= daily_fdd
                if water_heat_bank < 0:
                    overflow = -water_heat_bank
                    water_heat_bank = 0
                    cumulative_fdd += overflow
                    current_thickness = alpha * np.sqrt(cumulative_fdd)
                water_status.append('Cooling')
            else:
                # Grow ice
                cumulative_fdd += daily_fdd
                current_thickness = alpha * np.sqrt(cumulative_fdd)
                water_status.append('Freezing')
        else:
            # Base melt from temperature
            melt_degrees = avg_temp - base_temp
            effective_melt = melt_degrees * base_melt

            # Solar enhancement: stronger sun = faster melt
            if use_solar:
                solar_melt = s * 0.05 * max(0, 40 - avg_temp) / 10
                effective_melt += solar_melt

            # Rain melt (liquid water transfers heat very efficiently)
            if p > 0:
                rain_melt = min(p * 0.01, 0.3)  # Cap rain effect
                effective_melt += rain_melt

            # Flow enhances melting
            effective_melt *= (flow ** 0.3)

            if current_thickness > 0:
                current_thickness -= effective_melt
                if current_thickness < 0:
                    current_thickness = 0
                    cumulative_fdd = 0
                else:
                    cumulative_fdd = (current_thickness / alpha) ** 2
                water_status.append('Thawing')
            else:
                # Recharge heat bank
                recharge = melt_degrees * HEAT_BANK_RECHARGE_RATE
                water_heat_bank = min(water_heat_bank + recharge, MAX_HEAT_BANK)
                water_status.append('Warming')

        ice_thickness[i] = current_thickness

        if current_thickness >= ICE_CLOSURE_THRESHOLD:
            is_closed[i] = 1

    df['Closed'] = is_closed
    df['Metric'] = ice_thickness
    df['IceThickness'] = ice_thickness
    df['WaterStatus'] = water_status
    return df


def calculate_empirical_model(df, threshold=25, days_freeze=3, days_thaw=2, flow_df=None):
    """
    Simple Consecutive Days Model (Original).
    Best for quick estimates and explaining to non-technical stakeholders.
    """
    df = df.copy()
    temps = df['tmax_f'].values
    is_closed = np.zeros(len(df))
    current_state = 0
    counter = 0

    for i in range(len(df)):
        temp = temps[i]
        if current_state == 0:
            if temp < threshold:
                counter += 1
            else:
                counter = 0
            if counter >= days_freeze:
                current_state = 1
                counter = 0
        else:
            is_closed[i] = 1
            if temp > threshold:
                counter += 1
            else:
                counter = 0
            if counter >= days_thaw:
                current_state = 0
                counter = 0

    df['Closed'] = is_closed
    df['Metric'] = temps
    df['IceThickness'] = np.nan  # Not calculable with this model
    return df


def calculate_ensemble_model(df, flow_df=None, weights=None,
                             initial_heat_bank=DEFAULT_HEAT_BANK):
    """
    Ensemble Model - Weighted average of multiple models.

    Combines predictions from:
    - Stefan Equation (physics-based)
    - Modified Degree Day (environmental corrections)
    - Linear FDD (simple baseline)

    Default weights favor the more sophisticated models.
    """
    if weights is None:
        weights = {
            'stefan': 0.4,
            'mdd': 0.4,
            'linear': 0.2
        }

    # Run all models with heat bank
    stefan_df = calculate_stefan_model(df.copy(), flow_df=flow_df,
                                       initial_heat_bank=initial_heat_bank)
    mdd_df = calculate_modified_degree_day(df.copy(), flow_df=flow_df,
                                           initial_heat_bank=initial_heat_bank)
    linear_df = calculate_linear_fdd_model(df.copy(), fdd_limit=65, melt_factor=3.0,
                                           base_temp=32, flow_df=flow_df)

    df = df.copy()

    # Weighted average of ice thickness
    df['IceThickness'] = (
        weights['stefan'] * stefan_df['IceThickness'] +
        weights['mdd'] * mdd_df['IceThickness'] +
        weights['linear'] * linear_df['IceThickness']
    )

    df['Metric'] = df['IceThickness']

    # Use MDD water status as the ensemble status (most comprehensive model)
    df['WaterStatus'] = mdd_df['WaterStatus']

    # Closure if ANY model predicts closure (conservative) or majority (balanced)
    # Using weighted probability approach
    closure_prob = (
        weights['stefan'] * stefan_df['Closed'] +
        weights['mdd'] * mdd_df['Closed'] +
        weights['linear'] * linear_df['Closed']
    )
    df['Closed'] = (closure_prob >= 0.5).astype(int)
    df['ClosureProbability'] = closure_prob

    return df


def calculate_canadian_ice_model(df, base_temp=32, k=0.027, flow_df=None,
                                  initial_heat_bank=DEFAULT_HEAT_BANK):
    """
    Canadian Ice Thickness Model (Ashton, 1989) with thermal lag.

    Used by Canadian Coast Guard for ice road certification.

    h = k × √(S)

    Where:
    - h = ice thickness (cm, converted to inches)
    - k = empirical coefficient (0.027 for moderate snow cover)
    - S = accumulated frost degree-days (Celsius)

    This model has been validated against actual ice measurements
    on northern rivers and lakes.
    """
    df = df.copy()

    # Convert to Celsius for this model
    temps_c = (df['tavg_f'].values - 32) * 5/9
    base_c = (base_temp - 32) * 5/9

    flow_factors = np.ones(len(df))
    if flow_df is not None and not flow_df.empty:
        for i, idx in enumerate(df.index):
            if idx in flow_df.index:
                flow_factors[i] = flow_df.loc[idx, 'flow_factor']

    ice_thickness = np.zeros(len(df))
    is_closed = np.zeros(len(df))
    water_status = []
    accumulated_fdd = 0
    current_thickness = 0.0
    # Convert heat bank to Celsius degree-days
    water_heat_bank = initial_heat_bank * 5/9

    for i in range(len(df)):
        temp_c = temps_c[i]
        flow = flow_factors[i]

        if temp_c < base_c:
            daily_fdd = base_c - temp_c
            daily_fdd /= (flow ** 0.5)

            if water_heat_bank > 0:
                water_heat_bank -= daily_fdd
                if water_heat_bank < 0:
                    overflow = -water_heat_bank
                    water_heat_bank = 0
                    accumulated_fdd += overflow
                    thickness_cm = k * np.sqrt(accumulated_fdd * 100)
                    current_thickness = thickness_cm / 2.54
                water_status.append('Cooling')
            else:
                accumulated_fdd += daily_fdd
                thickness_cm = k * np.sqrt(accumulated_fdd * 100)
                current_thickness = thickness_cm / 2.54
                water_status.append('Freezing')
        else:
            melt_c = temp_c - base_c
            melt_inches = melt_c * 0.15 * (flow ** 0.3)

            if current_thickness > 0:
                current_thickness -= melt_inches
                if current_thickness < 0:
                    current_thickness = 0
                    accumulated_fdd = 0
                else:
                    thickness_cm = current_thickness * 2.54
                    accumulated_fdd = (thickness_cm / k) ** 2 / 100
                water_status.append('Thawing')
            else:
                recharge = melt_c * HEAT_BANK_RECHARGE_RATE
                max_bank_c = MAX_HEAT_BANK * 5/9
                water_heat_bank = min(water_heat_bank + recharge, max_bank_c)
                water_status.append('Warming')

        ice_thickness[i] = current_thickness

        if current_thickness >= ICE_CLOSURE_THRESHOLD:
            is_closed[i] = 1

    df['Closed'] = is_closed
    df['Metric'] = ice_thickness
    df['IceThickness'] = ice_thickness
    df['WaterStatus'] = water_status
    return df

# --- 3. UI SIDEBAR & CONTROLS ---

st.sidebar.title("Genesee Ice Tool")

# A. Scenario Presets
st.sidebar.subheader("1. Quick Scenarios")
scenario = st.sidebar.radio(
    "Select a Risk Profile:",
    ["Optimistic (Fast Melt)", "Realistic (Standard)", "Conservative (Safety First)"],
    index=1,
    help="Presets that adjust the physics sliders below."
)

# Apply Presets (tuned for river conditions, not lakes)
if scenario == "Optimistic (Fast Melt)":
    def_alpha = 0.60  # River turbulence slows ice growth
    def_melt = 0.15
    def_base = 31  # Turbulence delays freezing
elif scenario == "Conservative (Safety First)":
    def_alpha = 0.85  # Calmer conditions, closer to lake behavior
    def_melt = 0.08
    def_base = 32
else:  # Realistic
    def_alpha = 0.78  # River coefficient (vs 0.9 for calm lakes)
    def_melt = 0.12
    def_base = 32

st.sidebar.markdown("---")

# B. Model Selection
st.sidebar.subheader("2. Ice Model Selection")
MODEL_OPTIONS = {
    "Modified Degree Day (Recommended)": "mdd",
    "Stefan Equation": "stefan",
    "Canadian Coast Guard Model": "canadian",
    "Ensemble (Multi-Model Average)": "ensemble",
    "Linear FDD (Legacy)": "linear",
    "Simple Rule-Based": "simple"
}

model_display = st.sidebar.selectbox(
    "Select Ice Formation Model",
    list(MODEL_OPTIONS.keys()),
    index=0,  # Default to MDD (most comprehensive for rivers)
    help="Different approaches to modeling ice thickness."
)
model_type = MODEL_OPTIONS[model_display]

# Model descriptions
model_descriptions = {
    "stefan": """
    **Stefan Equation**: Physics-accurate square-root ice growth
    with thermal lag. Includes:
    - Water heat bank (cooling delay before freeze)
    - Diminishing ice growth rate
    - River flow adjustment
    """,
    "mdd": """
    **Modified Degree Day** (Recommended): Most comprehensive
    model with corrections for:
    - Water heat bank (thermal lag)
    - Wind speed (disrupts ice formation)
    - Solar radiation (Feb sun melts at 30°F)
    - Snow insulation (slows ice growth)
    - River flow/turbulence
    """,
    "canadian": """
    **Canadian Ice Model** (Ashton, 1989): Coast Guard
    certification standard with thermal lag. Validated on
    northern rivers and lakes.
    """,
    "ensemble": """
    **Ensemble Model**: Weighted average of Stefan, MDD, and
    Linear models. Reduces individual model bias by combining
    multiple approaches.
    """,
    "linear": """
    **Linear FDD**: Legacy simple model. Ice grows/melts
    linearly with temperature. No thermal lag - less accurate
    but easy to understand.
    """,
    "simple": """
    **Simple Rule-Based**: "N days below X°F = closed."
    Best for quick estimates and explaining to non-technical
    stakeholders.
    """
}
st.sidebar.markdown(model_descriptions[model_type])

st.sidebar.markdown("---")

# C. Advanced Controls
st.sidebar.subheader("3. Model Parameters")

# Data options
use_usgs_flow = st.sidebar.checkbox(
    "Include USGS River Flow Data",
    value=True,
    help="Use actual Genesee River discharge to adjust ice formation/melt rates."
)

if model_type in ["stefan", "mdd", "canadian", "ensemble"]:
    if model_type != "ensemble":
        alpha = st.sidebar.slider(
            "Stefan Coefficient (α)",
            0.5, 1.2, def_alpha, 0.05,
            help="Ice growth rate. Lower = slower freezing (turbulent water). "
                 "Typical: 0.8-1.0 for calm freshwater."
        )
        melt_rate = st.sidebar.slider(
            "Melt Rate",
            0.05, 0.25, def_melt, 0.01,
            help="Inches of ice lost per degree-day above freezing."
        )
        base_temp = st.sidebar.slider(
            "Effective Freezing Point (°F)",
            28, 33, def_base,
            help="Temperature at which ice begins forming. "
                 "Lower values simulate high-flow years."
        )

    if model_type == "mdd":
        st.sidebar.markdown("**Environmental Factors:**")
        use_wind = st.sidebar.checkbox("Include Wind Effects", value=True)
        use_solar = st.sidebar.checkbox("Include Solar Radiation", value=True)
        use_snow = st.sidebar.checkbox("Include Snow Insulation", value=True,
                                       help="Snow on ice slows further ice growth.")

    if model_type == "ensemble":
        st.sidebar.markdown("**Model Weights:**")
        w_stefan = st.sidebar.slider("Stefan Weight", 0.0, 1.0, 0.4, 0.1)
        w_mdd = st.sidebar.slider("MDD Weight", 0.0, 1.0, 0.4, 0.1)
        w_linear = st.sidebar.slider("Linear Weight", 0.0, 1.0, 0.2, 0.1)
        # Normalize weights
        total_w = w_stefan + w_mdd + w_linear
        if total_w > 0:
            weights = {
                'stefan': w_stefan / total_w,
                'mdd': w_mdd / total_w,
                'linear': w_linear / total_w
            }
        else:
            weights = {'stefan': 0.4, 'mdd': 0.4, 'linear': 0.2}

elif model_type == "linear":
    fdd_threshold = st.sidebar.slider(
        "Ice Limit (FDD)", 30, 120, 65,
        help="Cumulative freezing degree-days to trigger closure."
    )
    melt_factor = st.sidebar.slider(
        "Melt Multiplier", 1.0, 5.0, 3.0, 0.5,
        help="How much faster melting occurs vs freezing."
    )
    base_temp = st.sidebar.slider(
        "Freezing Point (°F)", 28, 33, 32
    )

else:  # simple
    simple_threshold = st.sidebar.slider("Max Temp Threshold (°F)", 15, 35, 25)
    days_to_freeze = st.sidebar.slider("Days to Close", 1, 10, 3)
    days_to_thaw = st.sidebar.slider("Days to Reopen", 1, 10, 2)

st.sidebar.markdown("---")
st.sidebar.subheader("4. Data Range")
start_year = st.sidebar.selectbox("Data Start Year", [1980, 1990, 2000, 2010], index=2)


# --- 4. EXECUTION ---
raw_df = get_rochester_data(start_year)
if raw_df.empty:
    st.error("Could not fetch weather data from Meteostat.")
    st.stop()

# Fetch USGS flow data if requested
flow_df = None
if use_usgs_flow:
    with st.spinner("Fetching USGS river flow data..."):
        flow_df = get_usgs_flow_data(start_year)
    if flow_df.empty:
        st.sidebar.warning("USGS flow data unavailable. Using default flow assumptions.")
        flow_df = None

# Execute selected model
if model_type == "stefan":
    df = calculate_stefan_model(
        raw_df, alpha=alpha, melt_rate=melt_rate,
        base_temp=base_temp, flow_df=flow_df
    )
    metric_label = "Ice Thickness (inches)"
    threshold_val = ICE_CLOSURE_THRESHOLD

elif model_type == "mdd":
    df = calculate_modified_degree_day(
        raw_df, alpha=alpha, base_melt=melt_rate,
        base_temp=base_temp, use_wind=use_wind,
        use_solar=use_solar, use_snow=use_snow,
        flow_df=flow_df
    )
    metric_label = "Ice Thickness (inches)"
    threshold_val = ICE_CLOSURE_THRESHOLD

elif model_type == "canadian":
    df = calculate_canadian_ice_model(
        raw_df, base_temp=base_temp, k=0.027 * (alpha / 0.9),
        flow_df=flow_df
    )
    metric_label = "Ice Thickness (inches)"
    threshold_val = ICE_CLOSURE_THRESHOLD

elif model_type == "ensemble":
    df = calculate_ensemble_model(raw_df, flow_df=flow_df, weights=weights)
    metric_label = "Ice Thickness (inches)"
    threshold_val = ICE_CLOSURE_THRESHOLD

elif model_type == "linear":
    df = calculate_linear_fdd_model(
        raw_df, fdd_limit=fdd_threshold,
        melt_factor=melt_factor, base_temp=base_temp,
        flow_df=flow_df
    )
    metric_label = "Ice Severity Score (FDD)"
    threshold_val = fdd_threshold

else:  # simple
    df = calculate_empirical_model(
        raw_df, threshold=simple_threshold,
        days_freeze=days_to_freeze, days_thaw=days_to_thaw,
        flow_df=flow_df
    )
    metric_label = "Daily High Temp (°F)"
    threshold_val = simple_threshold

# --- 5. DASHBOARD HEADER ---

# Calculate High-Level Metrics
total_days = len(df)
closed_days = df['Closed'].sum()
uptime_pct = 100 - (closed_days / total_days * 100)

# Winter Specific Metrics (Dec-Mar)
df['Month'] = df.index.month
winter_df = df[df['Month'].isin([12, 1, 2, 3])]
winter_total = len(winter_df)
winter_closed = winter_df['Closed'].sum()
winter_uptime = 100 - (winter_closed / winter_total * 100) if winter_total > 0 else 100

yearly_closures = df.groupby(df.index.year)['Closed'].sum() / 7  # Weeks
avg_weeks_closed = yearly_closures.mean()

# Max ice thickness if available
if 'IceThickness' in df.columns and not df['IceThickness'].isna().all():
    max_ice = df['IceThickness'].max()
    avg_max_ice_per_year = df.groupby(df.index.year)['IceThickness'].max().mean()
else:
    max_ice = None
    avg_max_ice_per_year = None

st.title("Genesee River Navigability Dashboard")

# Model info bar
col_model, col_scenario, col_flow, col_water = st.columns(4)
col_model.markdown(f"**Model:** {model_display}")
col_scenario.markdown(f"**Scenario:** {scenario}")
if flow_df is not None and not flow_df.empty:
    col_flow.markdown("**Flow Data:** USGS Active")
else:
    col_flow.markdown("**Flow Data:** Not Available")

# Water temperature status indicator
if 'WaterStatus' in df.columns:
    # Get recent status (last 7 days)
    recent_status = df['WaterStatus'].tail(7)
    status_counts = recent_status.value_counts()
    dominant_status = status_counts.index[0] if len(status_counts) > 0 else "Unknown"

    # Color-code the status
    status_colors = {
        'Warming': '🟢',
        'Cooling': '🟡',
        'Freezing': '🔵',
        'Thawing': '🟠'
    }
    status_icon = status_colors.get(dominant_status, '⚪')
    col_water.markdown(f"**Water Status:** {status_icon} {dominant_status}")
else:
    col_water.markdown("**Water Status:** N/A")

st.markdown("---")

# KPI Cards - Two rows
k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Annual Avg Closure",
    f"{avg_weeks_closed:.1f} Weeks",
    delta_color="inverse"
)
k2.metric(
    "Worst Year",
    f"{yearly_closures.max():.1f} Weeks",
    f"({int(yearly_closures.idxmax())})",
    delta_color="inverse"
)
k3.metric(
    "Winter Uptime (Dec-Mar)",
    f"{winter_uptime:.1f}%",
    "Reliability"
)
k4.metric(
    "Total Data Points",
    f"{len(df):,}",
    f"Since {start_year}"
)

# Second row of metrics if ice thickness is available
if max_ice is not None:
    k5, k6, k7, k8 = st.columns(4)
    k5.metric(
        "Max Ice Recorded",
        f"{max_ice:.1f}\"",
        "Historical Peak"
    )
    k6.metric(
        "Avg Annual Max Ice",
        f"{avg_max_ice_per_year:.1f}\"",
        "Typical Winter"
    )
    k7.metric(
        "Closure Threshold",
        f"{ICE_CLOSURE_THRESHOLD}\"",
        "Safety Limit"
    )
    # Calculate average first closure date in winter
    winter_closures = df[df['Closed'] == 1]
    if len(winter_closures) > 0:
        first_closure_by_year = winter_closures.groupby(winter_closures.index.year).apply(
            lambda x: x.index.min()
        )
        avg_first_closure_day = first_closure_by_year.dt.dayofyear.mean()
        # Convert day of year to approximate date
        if avg_first_closure_day > 0:
            from datetime import timedelta
            approx_date = datetime(2000, 1, 1) + timedelta(days=int(avg_first_closure_day) - 1)
            k8.metric(
                "Avg First Closure",
                approx_date.strftime("%b %d"),
                "Typical Start"
            )
        else:
            k8.metric("Avg First Closure", "N/A", "")
    else:
        k8.metric("Avg First Closure", "N/A", "No closures")


# --- 6. TABS & CHARTS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Timeline Analysis",
    "Seasonality Heatmap",
    "Annual Comparison",
    "Model Comparison"
])

# --- TAB 1: TIMELINE ---
with tab1:
    st.subheader("Historical Timeline with Closure Zones")
    st.markdown("Zoom in to see specific events. **Red Backgrounds** indicate when the river is impassable.")
    
    # Filter for performance
    recent_years = 5
    cutoff_date = datetime.now().year - recent_years
    plot_df = df[df.index.year >= cutoff_date]
    
    fig_time = go.Figure()
    
    # The Metric Line
    fig_time.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df['Metric'],
        mode='lines', name=metric_label,
        line=dict(color='cornflowerblue', width=2),
        fill='tozeroy' if "Physics" in model_type else None
    ))
    
    # Add Threshold Line
    fig_time.add_hline(y=threshold_val, line_dash="dash", line_color="black", annotation_text="Threshold")
    
    # Add Red Background Rectangles for Closures
    # We find start/end of closure blocks to draw rectangles
    closure_blocks = []
    is_currently_closed = False
    start_block = None
    
    # Simple logic to find blocks (loop is fast enough for 5 years of data)
    for date, row in plot_df.iterrows():
        if row['Closed'] == 1 and not is_currently_closed:
            is_currently_closed = True
            start_block = date
        elif row['Closed'] == 0 and is_currently_closed:
            is_currently_closed = False
            closure_blocks.append((start_block, date))
            
    # Add shapes
    shapes = []
    for start, end in closure_blocks:
        shapes.append(dict(
            type="rect", xref="x", yref="paper",
            x0=start, x1=end, y0=0, y1=1,
            fillcolor="red", opacity=0.2, layer="below", line_width=0
        ))
    
    fig_time.update_layout(shapes=shapes, height=450, hovermode="x unified", yaxis_title=metric_label)
    st.plotly_chart(fig_time, use_container_width=True)

# --- TAB 2: HEATMAP ---
with tab2:
    st.subheader("Closure Probability by Month")
    st.markdown("Darker colors indicate higher frequency of closures.")
    
    # Prepare Data for Heatmap: Year vs Month
    heatmap_data = df.copy()
    heatmap_data['Year'] = heatmap_data.index.year
    heatmap_data['Month'] = heatmap_data.index.month_name()
    
    # Aggregate closures per month/year
    # We only care about winter months usually
    month_order = ['November', 'December', 'January', 'February', 'March', 'April']
    heatmap_data = heatmap_data[heatmap_data['Month'].isin(month_order)]
    
    pivot = heatmap_data.groupby(['Year', 'Month'])['Closed'].sum().unstack()
    pivot = pivot.reindex(columns=month_order)
    pivot = pivot.fillna(0)
    
    fig_heat = px.imshow(
        pivot, 
        labels=dict(x="Month", y="Year", color="Days Closed"),
        x=month_order,
        color_continuous_scale="Reds",
        aspect="auto"
    )
    fig_heat.update_layout(height=600)
    st.plotly_chart(fig_heat, use_container_width=True)

# --- TAB 3: ANNUAL COMPARISON ---
with tab3:
    st.subheader("Worst Winter Ranking")
    
    # Bar Chart of weeks closed
    annual_weeks = df.groupby(df.index.year)['Closed'].sum() / 7
    
    # Color logic
    colors = ['green' if x < 2 else 'orange' if x < 5 else 'red' for x in annual_weeks]
    
    fig_bar = go.Figure(go.Bar(
        x=annual_weeks.index,
        y=annual_weeks.values,
        marker_color=colors,
        text=[f"{x:.1f} wks" for x in annual_weeks.values],
        textposition='auto'
    ))
    
    fig_bar.update_layout(
        xaxis_title="Year", yaxis_title="Weeks Closed",
        height=500
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Data Table
    display_cols = ['tavg_f', 'Closed', 'Metric']
    if 'IceThickness' in df.columns:
        display_cols.append('IceThickness')
    st.markdown("### Raw Data")
    st.dataframe(df[display_cols].sort_index(ascending=False).head(1000), height=300)

# --- TAB 4: MODEL COMPARISON ---
with tab4:
    st.subheader("Compare All Models")
    st.markdown("""
    Run all ice models on the same dataset to see how predictions differ.
    This helps identify model uncertainty and choose the most appropriate approach.
    """)

    if st.button("Run Model Comparison"):
        with st.spinner("Running all models..."):
            # Run each model with heat bank for thermal lag
            comparison_results = {}

            # Stefan (with heat bank)
            stefan_df = calculate_stefan_model(raw_df.copy(), flow_df=flow_df,
                                               initial_heat_bank=DEFAULT_HEAT_BANK)
            comparison_results['Stefan Equation'] = stefan_df

            # MDD (with heat bank)
            mdd_df = calculate_modified_degree_day(raw_df.copy(), flow_df=flow_df,
                                                   initial_heat_bank=DEFAULT_HEAT_BANK)
            comparison_results['Modified Degree Day'] = mdd_df

            # Canadian (with heat bank)
            canadian_df = calculate_canadian_ice_model(raw_df.copy(), flow_df=flow_df,
                                                       initial_heat_bank=DEFAULT_HEAT_BANK)
            comparison_results['Canadian Ice Model'] = canadian_df

            # Linear FDD (no heat bank - legacy model)
            linear_df = calculate_linear_fdd_model(
                raw_df.copy(), fdd_limit=65, melt_factor=3.0, base_temp=32, flow_df=flow_df
            )
            comparison_results['Linear FDD'] = linear_df

            # Ensemble (with heat bank)
            ensemble_df = calculate_ensemble_model(raw_df.copy(), flow_df=flow_df,
                                                   initial_heat_bank=DEFAULT_HEAT_BANK)
            comparison_results['Ensemble'] = ensemble_df

        # Create comparison metrics table
        st.markdown("### Summary Statistics")
        summary_data = []
        for name, result_df in comparison_results.items():
            yearly = result_df.groupby(result_df.index.year)['Closed'].sum() / 7
            summary_data.append({
                'Model': name,
                'Avg Weeks Closed/Year': f"{yearly.mean():.1f}",
                'Max Weeks (Worst Year)': f"{yearly.max():.1f}",
                'Min Weeks (Best Year)': f"{yearly.min():.1f}",
                'Std Dev': f"{yearly.std():.2f}",
                'Total Closure Days': int(result_df['Closed'].sum())
            })

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # Plot ice thickness comparison for recent winter
        st.markdown("### Ice Thickness Comparison (Last 3 Years)")

        # Get last 3 years of winter data
        cutoff = datetime.now().year - 3
        fig_compare = go.Figure()

        colors = {
            'Stefan Equation': 'blue',
            'Modified Degree Day': 'green',
            'Canadian Ice Model': 'orange',
            'Linear FDD': 'purple',
            'Ensemble': 'red'
        }

        for name, result_df in comparison_results.items():
            if 'IceThickness' in result_df.columns:
                plot_data = result_df[result_df.index.year >= cutoff]
                # Only show winter months
                plot_data = plot_data[plot_data.index.month.isin([11, 12, 1, 2, 3, 4])]
                fig_compare.add_trace(go.Scatter(
                    x=plot_data.index,
                    y=plot_data['IceThickness'],
                    mode='lines',
                    name=name,
                    line=dict(color=colors.get(name, 'gray'))
                ))

        fig_compare.add_hline(
            y=ICE_CLOSURE_THRESHOLD,
            line_dash="dash",
            line_color="black",
            annotation_text=f"Closure Threshold ({ICE_CLOSURE_THRESHOLD}\")"
        )

        fig_compare.update_layout(
            height=500,
            yaxis_title="Ice Thickness (inches)",
            xaxis_title="Date",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig_compare, use_container_width=True)

        # Agreement analysis
        st.markdown("### Model Agreement Analysis")
        st.markdown("""
        Shows when models agree or disagree on closure status.
        High disagreement indicates uncertainty - consider using conservative estimates.
        """)

        # Create agreement matrix for last 3 years
        agreement_df = pd.DataFrame(index=comparison_results['Stefan Equation'].index)
        for name, result_df in comparison_results.items():
            agreement_df[name] = result_df['Closed']

        agreement_df = agreement_df[agreement_df.index.year >= cutoff]

        # Calculate agreement percentage
        agreement_df['Models Predicting Closure'] = agreement_df.sum(axis=1)
        agreement_df['Agreement'] = agreement_df['Models Predicting Closure'].apply(
            lambda x: 'Full Agreement (Open)' if x == 0
            else 'Full Agreement (Closed)' if x == 5
            else 'Majority Closed' if x >= 3
            else 'Majority Open'
        )

        # Show agreement distribution
        agreement_counts = agreement_df['Agreement'].value_counts()
        col1, col2 = st.columns(2)

        with col1:
            fig_pie = px.pie(
                values=agreement_counts.values,
                names=agreement_counts.index,
                title="Prediction Agreement Distribution",
                color_discrete_sequence=['green', 'lightgreen', 'orange', 'red']
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Uncertainty days by month
            uncertain_days = agreement_df[
                agreement_df['Models Predicting Closure'].isin([1, 2, 3, 4])
            ]
            if len(uncertain_days) > 0:
                uncertain_by_month = uncertain_days.groupby(
                    uncertain_days.index.month
                ).size()
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                fig_uncertain = go.Figure(go.Bar(
                    x=[month_names[m-1] for m in uncertain_by_month.index],
                    y=uncertain_by_month.values,
                    marker_color='orange'
                ))
                fig_uncertain.update_layout(
                    title="Days with Model Disagreement by Month",
                    yaxis_title="Number of Days",
                    height=350
                )
                st.plotly_chart(fig_uncertain, use_container_width=True)

    else:
        st.info("Click 'Run Model Comparison' to compare all available ice models.")