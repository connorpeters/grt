# ⛴️ Genesee River Transit: Ice Risk Analyzer

### **Overview**

This application is a specialized decision-support tool designed for the **Genesee River Transit (GRT)** system in Rochester, NY. It estimates historical and future navigability of the Genesee River during winter months.

The primary goal is to answer: **"How many weeks per year will the river be impassable for aluminum-hull passenger boats due to ice?"**

### **The Problem**

Aluminum hulls are particularly vulnerable to sheet ice.

* **Safety Threshold:** Standard aluminum hulls cannot safely break ice thicker than **0.5 inches**.
* **Critical Failure:** Ice exceeding **2.0–3.0 inches** renders travel impossible and risks structural damage to the vessel.
* **The Challenge:** Measuring air temperature alone is insufficient. A single day of 33°F weather does not melt a 3-inch ice sheet. Ice has "thermal inertia."

---

### **Methodology: The Logic Engines**

This tool includes six distinct modeling approaches to estimate ice coverage, from simple heuristics to physics-based simulations.

#### **1. Stefan Equation Model**

* **Algorithm:** Square-root ice growth based on cumulative Freezing Degree Days (FDD) with thermal lag.
* **Formula:** `Ice Thickness = α × √(FDD)` where α ≈ 0.7-0.8 for rivers (lower than lakes due to turbulence).
* **Key Features:**
  * **Water Heat Bank:** Models the thermal mass of the water. The river must lose ~35 degree-days of heat before ice can form—this prevents unrealistic instant freezing on the first cold day (rivers cool faster than lakes).
  * **Diminishing Returns:** First 10 FDD produce more ice than the second 10 FDD (heat must conduct through existing ice).
  * **River Flow Adjustment:** USGS discharge data reduces ice formation during high-flow periods.

#### **2. Modified Degree Day Model (MDD) — Recommended**

* **Algorithm:** Stefan equation with environmental corrections and full thermal modeling.
* **Factors included:**
  * **Water Heat Bank:** ~35 degree-day thermal lag before freezing begins
  * **Wind speed:** High winds prevent sheet ice formation (waves break up forming ice)
  * **Solar radiation:** February sun melts ice even at 30°F (declination angle calculation)
  * **Snow insulation:** Snow cover on ice slows further growth (insulates from cold air)
  * **Precipitation:** Rain accelerates melt
  * **River flow:** USGS turbulence data
* **Use Case:** Most comprehensive and accurate model for river conditions.

#### **3. Canadian Coast Guard Model**

* **Algorithm:** Ashton (1989) empirical formula used for ice road certification.
* **Formula:** `h = k × √(S)` where k ≈ 0.027 for moderate snow cover.
* **Validation:** Field-tested on northern Canadian rivers and lakes.
* **Use Case:** Regulatory-grade predictions when conservative estimates are needed.

#### **4. Ensemble Model**

* **Algorithm:** Weighted average of Stefan, MDD, and Linear FDD models.
* **How it works:** Combines predictions to reduce individual model bias.
* **Closure Logic:** Predicts closure when ≥50% of weighted models agree.
* **Use Case:** When model uncertainty is high, provides balanced estimate.

#### **5. Linear FDD Model (Legacy)**

* **Algorithm:** Simple linear accumulation of freezing degree-days.
* **How it works:** Ice grows/melts linearly with temperature difference from 32°F.
* **Limitation:** Doesn't capture diminishing returns of ice growth physics.
* **Use Case:** Backward compatibility; easy to explain.

#### **6. Simple Rule-Based Model**

* **Algorithm:** Consecutive daily thresholds.
* **How it works:** "If max temperature stays below X°F for N consecutive days, river is closed."
* **Use Case:** Quick estimates; explaining to non-technical stakeholders.

---

### **Key Parameters & Defaults**

The application allows users to toggle "Scenarios" (Optimistic, Realistic, Conservative). These scenarios adjust the underlying physics parameters as follows:

| Parameter | Default (Realistic) | Range | Explanation |
| --- | --- | --- | --- |
| **Stefan Coefficient (α)** | **0.78** | 0.5 - 1.2 | Ice growth rate constant. Rivers use ~0.7-0.8 (turbulence); calm lakes use ~0.9. |
| **Melt Rate** | **0.12 in/°F** | 0.05 - 0.25 | Inches of ice lost per degree-day above freezing. Higher = faster spring melt. |
| **Base Temp** | **32°F** | 28 - 33 | Effective freezing point. Lower values simulate high-flow conditions where turbulence delays freeze. |
| **Water Heat Bank** | **35 °F-days** | Internal | Thermal mass of river water. Must be depleted before ice forms (~3-5 cold days). Rivers cool faster than lakes. |
| **Ice Closure Threshold** | **2.5 inches** | Fixed | Aluminum hull safety limit. River considered impassable above this thickness. |

### **Data Sources**

| Source | Data | Usage |
| --- | --- | --- |
| **Meteostat (KROC)** | Temperature, wind, precipitation, snow | Primary weather inputs for all models |
| **USGS Station 04231600** | River discharge (cfs) | Flow factor to adjust ice formation rate |

The USGS flow data provides real-time turbulence estimates. High discharge = more turbulence = harder for ice to form. The flow factor normalizes against median historical discharge.

---

### **How to Run**

**Prerequisites:**

* Python 3.8+
* Streamlit
* Meteostat (for historical weather data)
* Plotly

**Installation:**

```bash
pip install streamlit pandas numpy plotly meteostat requests
```

**Launch the Dashboard:**

```bash
streamlit run temp_analysis.py

```

---

### **Data Sources**

**Weather Data:**
* **Station:** KROC (Greater Rochester International Airport)
* **Library:** [Meteostat](https://meteostat.net/) Python library
* **Variables:** Temperature (tavg, tmin, tmax), wind speed, precipitation, snow
* **Fallback:** Calculates averages from tmin/tmax when tavg unavailable

**River Discharge Data:**
* **Station:** USGS 04231600 (Genesee River at Ford St, Rochester)
* **API:** [USGS Water Services](https://waterservices.usgs.gov/)
* **Variable:** Discharge in cubic feet per second (parameter 00060)
* **Purpose:** Adjusts ice formation rate based on actual river turbulence

### **Disclaimer**

*This tool provides probabilistic estimates based on historical atmospheric data. It does not account for physical river obstructions (log jams), water level management by the Mount Morris Dam, or specific localized industrial discharge that may warm the water.*